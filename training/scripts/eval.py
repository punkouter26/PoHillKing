"""Pass-bar evaluation for a checkpoint. Deterministic policy, domain randomization on, obs noise on.
  uv run python scripts/eval.py --rung r0 --ckpt runs/r1/r1_a/model_1000.pt [--seeds 10 --num-envs 256]

Disturbances for r0/r1: 2 m/s pelvis kicks every 5 s and 2 kg boxes at 6 m/s every 3-8 s.
r0  stand (zero command).            Pass: >= 90 % of impacts survived, every seed.
r1  random velocity commands.        Pass: r0 survival + tracking error < 0.15 m/s and < 0.2 rad/s.
    Tracking error = |command - 1 s moving average of body velocity|, counted only when the last impact,
    command change and reset are all more than 1.5 s old (a 2 m/s kick is itself a 2 m/s tracking error).
r2  arena, spawn r in [0, 2.6] m.    Pass: >= 90 % of robots spawned on the plateau (r < 1.4) are inside r < 1.2 for
    the last 10 s without falling, and >= 80 % of robots spawned on the slope (r > 2.3) reach r < 1.2 and stay.
r3  arena, two robots.               Pass: >= 90 % of pairs get within 0.6 m in 6 s, nobody leaves the plateau.
r5ffa  four robots, learner a_ vs three frozen opponents (--opponent). Pass: the learner is last one in in >= 40 %
    of rounds (25 % = no better than its opponents).
kim_r0b / kim_r1b are judged as r0 / r1 (KOTH_ROBOT=kim), with kicks scaled to the same impulse as on the G1
(--push-vel defaults to 2 m/s x m_G1 / m_Kim).
--leg-contact-rule unity: own foot-shin contact does not end a round (Unity / duel_stats.py); "train" keeps the rung's
training rule (on for every rung up to r4*, off for the staged rungs).
Prints one JSON line per seed plus a verdict; exit code 0 on pass, 1 on fail.
Duel and free-for-all rows also carry "flags" (reported, never part of the pass): stall (more than 30 % of rounds reach
the bell: trump_bull1/2, xg2), contact_avoidance (in touch, < 0.7 m apart, for under 15 % of the open round time: the
bull runs learned to stay away), self_ejection (more losses than wins against a standing opponent: r4att). A brain that
passes with a flag still deserves a look."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rsl_rl.runners import OnPolicyRunner
from koth.env import HUMANS, KothEnv, default_cfg
from koth.rules import OUT_RADIUS, PLATEAU_R
from scripts.train import train_cfg

DUEL_RUNGS = ("r4probe", "r4att", "r4league")
FFA_RUNGS = ("r5ffa",)


def load_policy(env, ckpt):
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    width = ck["actor_state_dict"]["obs_normalizer._mean"].shape[-1]
    # rebuilt from a Unity ONNX (scripts/onnx_to_ckpt.py, actor only), or a 112-input brain judged in a 121-input rung
    # (a fighter as the baseline of a free-for-all brain): the deterministic actor on the first columns it knows
    if not ck.get("critic_state_dict") or width < env.get_observations()["policy"].shape[-1]:
        from koth.actor import Actor
        actor = Actor(ck["actor_state_dict"]).cuda().eval()
        return lambda obs: actor(obs["policy"][:, :actor.obs_dim])   # a 112-input brain reads the first 112 of a 121-wide observation
    runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cuda")
    runner.load(ckpt)
    policy = runner.alg.get_policy(); policy.eval()
    return policy


def base_rung(rung: str) -> str:
    """kim_r0b -> r0, trump_r4bull -> r4league, trump_vs_g1 -> r4league, kim_ffa / kim_hill -> r5ffa: the bar a rung is judged on."""
    if "_vs_" in rung: return "r4league"
    if "_att_" in rung: return "r4att"
    body, _, rest = rung.partition("_")
    if body not in HUMANS: return rung
    if rest in ("ffa", "hill"): return "r5ffa"
    return {"r4bull": "r4league", "bull": "r4league"}.get(rest, rest[:2])


def eval_flat(env, policy, rung, steps):
    M = env.num_envs; dt = env.ctrl_dt; dev = env.device
    obs = env.reset()
    falls = torch.zeros((), dtype=torch.long, device=dev); impacts = torch.zeros((), dtype=torch.long, device=dev)
    ever_fell = torch.zeros(M, dtype=torch.bool, device=dev)
    ema = torch.zeros(M, 3, device=dev); quiet = torch.zeros(M, device=dev); prev_cmd = env.command.clone()
    lin_sum = torch.zeros((), dtype=torch.float64, device=dev); ang_sum = torch.zeros_like(lin_sum); cnt = torch.zeros_like(falls)
    for _ in range(steps):
        push_due = env.push_timer - dt <= 0; box_w = (env.proj_timer - dt <= 0) & env.cfg["projectile"]["enabled"]; box_due = box_w.repeat_interleave(env.A)
        with torch.no_grad():
            act = policy(obs)
        obs, _, done, extras = env.step(act)
        impacts += push_due.sum() + box_w.sum()
        falls += extras["fallen"].sum(); ever_fell |= extras["fallen"]
        o = obs["critic"]; v = torch.cat([o[:, 0:2], o[:, 5:6]], 1)
        ema += (v - ema) * (dt / 1.0)
        changed = (env.command - prev_cmd).abs().sum(1) > 1e-6; prev_cmd = env.command.clone()
        quiet = torch.where(push_due | box_due | done | changed, torch.zeros_like(quiet), quiet + dt)
        ema = torch.where(done[:, None], torch.zeros_like(ema), ema)
        ok = quiet > 1.5
        lin_sum += torch.where(ok, (env.command[:, :2] - ema[:, :2]).norm(dim=1), 0.0).sum().double()
        ang_sum += torch.where(ok, (env.command[:, 2] - ema[:, 2]).abs(), 0.0).sum().double(); cnt += ok.sum()
    falls, impacts, cnt, lin_sum, ang_sum = int(falls), int(impacts), int(cnt), float(lin_sum), float(ang_sum)    # one sync
    surv = 1 - falls / max(1, impacts)
    row = dict(impacts=impacts, falls=falls, impact_survival=round(surv, 4), envs_never_fell=round(1 - ever_fell.float().mean().item(), 4),
               lin_vel_err=round(lin_sum / max(1, cnt), 4), ang_vel_err=round(ang_sum / max(1, cnt), 4))
    row["pass"] = surv >= 0.9 and (rung == "r0" or (row["lin_vel_err"] < 0.15 and row["ang_vel_err"] < 0.2))
    return row


def eval_arena(env, policy, rung, steps):
    M = env.num_envs
    obs = env.reset()
    r0 = env._root()[0][:, :2].norm(dim=1).clone()
    alive = torch.ones(M, dtype=torch.bool, device=env.device)        # still in its first episode
    inside_late = torch.ones(M, dtype=torch.bool, device=env.device); left_plateau = torch.zeros(M, dtype=torch.bool, device=env.device)
    met = torch.zeros(M, dtype=torch.bool, device=env.device)
    for t in range(steps):
        with torch.no_grad():
            act = policy(obs)
        pos_before = env._root()[0][:, :2].clone()
        obs, _, done, extras = env.step(act)
        r = pos_before.norm(dim=1)
        if rung == "r3":
            p = pos_before.view(env.N, 2, 2); dist = (p[:, 0] - p[:, 1]).norm(dim=1).repeat_interleave(2)
            if t * env.ctrl_dt <= 6.0: met |= alive & (dist < 0.6)
            left_plateau |= alive & (r > PLATEAU_R)
        if t >= steps // 2: inside_late &= (r < 1.2) | ~alive
        fell_now = extras["fallen"] & alive
        inside_late &= ~fell_now
        alive &= ~done
    if rung == "r2":
        plat = r0 < 1.4; slope = r0 > 2.3
        hold = inside_late[plat].float().mean().item(); climb = inside_late[slope].float().mean().item() if slope.any() else 1.0
        return dict(n_plateau=int(plat.sum()), plateau_hold=round(hold, 4), n_slope=int(slope.sum()), slope_return=round(climb, 4),
                    all_return=round(inside_late.float().mean().item(), 4), **{"pass": hold >= 0.9 and climb >= 0.8})
    meet = met.view(env.N, 2).any(1).float().mean().item(); left = left_plateau.view(env.N, 2).any(1).float().mean().item()
    return dict(pairs=env.N, met_within_6s=round(meet, 4), pairs_with_self_ejection=round(left, 4), **{"pass": meet >= 0.9 and left <= 0.02})


def eval_duel(env, policy, steps, league=False):
    """Robot a attacks robot b. Counts, per world, the first of: b ejected (radius > 1.7 or fallen), a ejected/fallen."""
    N = env.N; obs = env.reset()
    open_ = torch.ones(N, dtype=torch.bool, device=env.device); win = torch.zeros_like(open_); lose = torch.zeros_like(open_)
    t_win = torch.full((N,), float('nan'), device=env.device); touch = torch.zeros((), device=env.device); open_t = torch.zeros((), device=env.device)
    jit = torch.zeros((), device=env.device)
    for t in range(steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, done, extras = env.step(act)
        sep = extras["sep"].view(N, 2)[:, 0]; touch += ((sep < 0.7) & open_).sum(); open_t += open_.sum()
        jit += (extras["jitter"].view(N, -1)[:, 0] * open_).sum()
        out = (extras["radius"] > OUT_RADIUS) | extras["fallen"]; out = out.view(N, 2)
        b_out = out[:, 1] & open_; a_out = out[:, 0] & open_ & ~b_out
        win |= b_out; lose |= a_out; t_win = torch.where(b_out, t * env.ctrl_dt, t_win)
        open_ &= ~(b_out | a_out | (done if done.shape[0] == N else done.view(N, 2)[:, 0]))
    w = win.float().mean().item(); l = lose.float().mean().item(); contact = float(touch / open_t.clamp(min=1))
    return dict(duels=N, attacker_ejects_defender=round(w, 4), attacker_lost=round(l, 4), undecided=round(1 - w - l, 4),
                median_time_s=round(float(t_win[win].median()) if win.any() else -1.0, 2), contact_share=round(contact, 4),
                jitter_rad=round(float(jit / open_t.clamp(min=1)), 5),
                flags=hack_flags(1 - w - l, contact, w, l, standing_opponent=not league),
                **{"pass": w >= 0.7 if not league else (w / max(w + l, 1e-9) > 0.55 and 1 - w - l < 0.15)})


def hack_flags(undecided, contact, w, l, standing_opponent=False) -> list:
    """Failure modes the run log has seen, as flags on an eval row (see the module docstring)."""
    f = []
    if undecided > 0.3: f.append("stall")
    if contact is not None and contact < 0.15: f.append("contact_avoidance")
    if standing_opponent and l > w: f.append("self_ejection")
    return f


def eval_ffa(env, policy, steps):
    """Free-for-all (> 2 robots): per world, the first of: the learner is last one in (win), the learner is out (loss)."""
    N = env.N; obs = env.reset()
    open_ = torch.ones(N, dtype=torch.bool, device=env.device); win = torch.zeros_like(open_); lose = torch.zeros_like(open_)
    t_win = torch.full((N,), float('nan'), device=env.device); jit = torch.zeros((), device=env.device); open_t = torch.zeros((), device=env.device)
    for t in range(steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, done, extras = env.step(act)
        jit += (extras["jitter"].view(N, -1)[:, 0] * open_).sum(); open_t += open_.sum()
        w_ = extras["world_won"] & open_; l_ = extras["world_lost"] & open_
        win |= w_; lose |= l_; t_win = torch.where(w_, t * env.ctrl_dt, t_win)
        open_ &= ~(w_ | l_ | done)
    w = win.float().mean().item(); l = lose.float().mean().item()
    return dict(rounds=N, learner_last_one_in=round(w, 4), learner_out=round(l, 4), undecided=round(1 - w - l, 4),
                median_time_s=round(float(t_win[win].median()) if win.any() else -1.0, 2), jitter_rad=round(float(jit / open_t.clamp(min=1)), 5), flags=hack_flags(1 - w - l, None, w, l), **{"pass": w >= 0.4})


def eval_cfg(rung, shove=False, rim_brake=None, slope=False, opponent=None, leg_contact_rule="train", push_vel=None, noise=True,
             combat2=False, decimation=None, fatigue="off", style_pose=None, ckpt=None):
    """The environment configuration eval.py judges a rung with (also used by contact_style.py). ckpt: the judged brain,
    whose action path (koth/action_map.py, from its run's config.json) the learner rows use; opponents use their own."""
    cfg = default_cfg(rung); base = base_rung(rung)
    if ckpt:
        from koth.action_map import ckpt_spec
        cfg["action"] = ckpt_spec(ckpt)
    if combat2: cfg["combat2"] = True
    cfg["duel_share"] = 0.0                  # a rung that trains duels and three-ways is judged on its three-ways; duels on the duel rungs
    if decimation: cfg["decimation"] = decimation
    if fatigue != "off": cfg["fatigue"].update(enabled=True, obs=fatigue == "obs")
    duel = base in DUEL_RUNGS
    cfg["shove"] = shove; cfg["style_ramp_iters"] = 0          # judged with the full style
    if style_pose: cfg["style_pose"] = style_pose
    if duel or base in FFA_RUNGS: cfg["max_radius"] = OUT_RADIUS
    if base in ("r4probe", "r4att"): cfg["spawn"].update(r=[[0.0, 0.5], [1.25, 1.35]])
    if rim_brake: cfg["goal_cmd"]["rim_brake"] = tuple(rim_brake)
    if rung == "r2" and slope: cfg["spawn"].update(r=[0.0, 2.6])
    if opponent: cfg["frozen_opponent"] = list(opponent)
    if base in ("r0", "r1"):
        v = 2.0 * cfg.get("mass_ratio", 1.0) if push_vel is None else push_vel
        cfg["push"] = dict(interval_s=[5.0, 5.0], vel=[v, v]); cfg["projectile"].update(speed=[6.0, 6.0])
    else:
        cfg["push"] = dict(interval_s=[1e6, 1e6], vel=[0.0, 0.0]); cfg["projectile"]["enabled"] = False
    if leg_contact_rule == "unity": cfg["terminate_on_leg_contact"] = False
    if not noise: cfg["noise"] = {k: 0.0 for k in cfg["noise"]}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--shove", action="store_true", help="the tested policy fights in the shove style (arm pose + action smoothing)")
    ap.add_argument("--style-pose", default=None, help="with --shove: the styled rest pose (bull: Trump's charge; default the rung's)")
    ap.add_argument("--rim-brake", type=float, nargs=3, default=None, help="rim gain vmin for the goal command")
    ap.add_argument("--slope", action="store_true", help="r2 only: also spawn robots on the slope (retired objective)")
    ap.add_argument("--opponent", default=None, nargs="+", help="frozen checkpoint(s) driving robot b_ (duel rungs) or b_..d_ (r5ffa)")
    ap.add_argument("--leg-contact-rule", choices=["train", "unity"], default="train",
                    help="unity: own foot-shin contact does not end a round, as in Unity and duel_stats.py")
    ap.add_argument("--push-vel", type=float, default=None, help="r0/r1 kick size in m/s (default 2, scaled by mass for Kim's rungs)")
    ap.add_argument("--combat2", action="store_true", help="the tested brain has the 121-input layout (second opponent block)")
    ap.add_argument("--decimation", type=int, default=None, help="physics steps per policy step, as the brain was trained")
    ap.add_argument("--fatigue", choices=["off", "on", "obs"], default="off", help="as the brain was trained")
    a = ap.parse_args()
    cfg = eval_cfg(a.rung, a.shove, a.rim_brake, a.slope, a.opponent, a.leg_contact_rule, a.push_vel, combat2=a.combat2,
                   decimation=a.decimation, fatigue=a.fatigue, style_pose=a.style_pose, ckpt=a.ckpt)
    base = base_rung(a.rung); duel = base in DUEL_RUNGS
    env = KothEnv(cfg, a.num_envs, seed=0)
    policy = load_policy(env, a.ckpt)
    steps = int(a.seconds / env.ctrl_dt); rows = []
    for seed in range(a.seeds):
        torch.manual_seed(1000 + seed)
        if duel: row = eval_duel(env, policy, steps, league=base == "r4league")
        elif base in FFA_RUNGS: row = eval_ffa(env, policy, steps)
        else: row = (eval_flat if base in ("r0", "r1") else eval_arena)(env, policy, base, steps)
        row = dict(seed=seed, **row); rows.append(row); print(json.dumps(row), flush=True)
    ok = all(r["pass"] for r in rows)
    print("VERDICT", a.rung, "PASS" if ok else "FAIL", f"({sum(r['pass'] for r in rows)}/{len(rows)} seeds)",
          *([] if a.leg_contact_rule == "train" else ["[leg-contact rule: unity]"]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
