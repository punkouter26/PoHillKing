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
Prints one JSON line per seed plus a verdict; exit code 0 on pass, 1 on fail."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rsl_rl.runners import OnPolicyRunner
from koth.env import KothEnv, default_cfg
from scripts.train import train_cfg


def load_policy(env, ckpt):
    runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cuda")
    runner.load(ckpt)
    policy = runner.alg.get_policy(); policy.eval()
    return policy


def eval_flat(env, policy, rung, steps):
    M = env.num_envs; dt = env.ctrl_dt
    obs = env.reset()
    falls = impacts = 0
    ever_fell = torch.zeros(M, dtype=torch.bool, device=env.device)
    ema = torch.zeros(M, 3, device=env.device); quiet = torch.zeros(M, device=env.device); prev_cmd = env.command.clone()
    lin_sum = ang_sum = cnt = 0.0
    for _ in range(steps):
        push_due = env.push_timer - dt <= 0; box_w = (env.proj_timer - dt <= 0) & env.cfg["projectile"]["enabled"]; box_due = box_w.repeat_interleave(env.A)
        with torch.no_grad():
            act = policy(obs)
        obs, _, done, extras = env.step(act)
        impacts += int(push_due.sum() + box_w.sum())
        falls += int(extras["fallen"].sum()); ever_fell |= extras["fallen"]
        o = obs["critic"]; v = torch.cat([o[:, 0:2], o[:, 5:6]], 1)
        ema += (v - ema) * (dt / 1.0)
        changed = (env.command - prev_cmd).abs().sum(1) > 1e-6; prev_cmd = env.command.clone()
        quiet = torch.where(push_due | box_due | done | changed, torch.zeros_like(quiet), quiet + dt)
        ema = torch.where(done[:, None], torch.zeros_like(ema), ema)
        ok = quiet > 1.5
        lin_sum += (env.command[:, :2] - ema[:, :2]).norm(dim=1)[ok].sum().item(); ang_sum += (env.command[:, 2] - ema[:, 2]).abs()[ok].sum().item(); cnt += int(ok.sum())
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
            left_plateau |= alive & (r > 1.5)
        if t >= steps // 2: inside_late &= (r < 1.2) | ~alive
        fell_now = extras["fallen"] & alive
        inside_late &= ~fell_now
        alive &= ~done
    if rung == "r2":
        plat = r0 < 1.4; slope = r0 > 2.3
        hold = inside_late[plat].float().mean().item(); climb = inside_late[slope].float().mean().item()
        return dict(n_plateau=int(plat.sum()), plateau_hold=round(hold, 4), n_slope=int(slope.sum()), slope_return=round(climb, 4),
                    all_return=round(inside_late.float().mean().item(), 4), **{"pass": hold >= 0.9 and climb >= 0.8})
    meet = met.view(env.N, 2).any(1).float().mean().item(); left = left_plateau.view(env.N, 2).any(1).float().mean().item()
    return dict(pairs=env.N, met_within_6s=round(meet, 4), pairs_with_self_ejection=round(left, 4), **{"pass": meet >= 0.9 and left <= 0.02})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    cfg = default_cfg(a.rung)
    if a.rung in ("r0", "r1"):
        cfg["push"] = dict(interval_s=[5.0, 5.0], vel=[2.0, 2.0]); cfg["projectile"].update(speed=[6.0, 6.0])
    else:
        cfg["push"] = dict(interval_s=[1e6, 1e6], vel=[0.0, 0.0]); cfg["projectile"]["enabled"] = False
    env = KothEnv(cfg, a.num_envs, seed=0)
    policy = load_policy(env, a.ckpt)
    steps = int(a.seconds / env.ctrl_dt); rows = []
    for seed in range(a.seeds):
        torch.manual_seed(1000 + seed)
        row = (eval_flat if a.rung in ("r0", "r1") else eval_arena)(env, policy, a.rung, steps)
        row = dict(seed=seed, **row); rows.append(row); print(json.dumps(row), flush=True)
    ok = all(r["pass"] for r in rows)
    print("VERDICT", a.rung, "PASS" if ok else "FAIL", f"({sum(r['pass'] for r in rows)}/{len(rows)} seeds)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
