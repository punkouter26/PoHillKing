"""Reward audit: where a brain's reward comes from on its own training rung, and the hacks the log has seen before.

  uv run python scripts/reward_audit.py --rung r4league --ckpt runs/league/careful1.pt --opponent runs/league/careful1.pt runs/league/gen7.pt
  bash scripts/reward_audit_all.sh                     # every shipped brain -> runs/audits/reward_<name>.json + reward_audit.md

Rollout on the training cfg (training noise and DR on, the brain's mean action), learner rows only. Per term: mean
reward per step, share of all positive / negative reward, and per-episode sums. Then:
  clip        env.py clips each step's shaped sum at zero (terminal terms outside). Share of steps where the clip binds,
              and the share of all penalty mass it erases: a penalty that only bites while the sum is positive is
              invisible exactly when the brain is doing badly.
  dominance   one shaped term carrying > 60 % of the positive mass (e.g. alive/upright paying more than the task).
  stall       duels and free-for-all: shaped return of a round that runs to the bell + the draw reward, against a decided
              round's shaped return + win. If the bell pays more, stalling is the better policy (run r4_a converged to
              exactly that; xg2 stalled with draw = loss).
  farm        a shaped term whose per-episode sum exceeds the win reward (push_out before its cap).
Writes --json; prints the flags. Exit code 0 always (it reports, it does not judge)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from koth.env import KothEnv, default_cfg

TERMINAL = ("termination", "win", "draw", "potential")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--opponent", nargs="+", default=None)
    ap.add_argument("--num-envs", type=int, default=128); ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--reward", nargs="*", default=[], help="the run's reward overrides (e.g. self_edge=-5 termination=-25)")
    ap.add_argument("--leg-contact-rule", choices=["train", "unity"], default="train"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cfg = default_cfg(a.rung)
    for kv in a.reward: k, v = kv.split("="); cfg["reward"][k] = float(v)
    if a.leg_contact_rule == "unity": cfg["terminate_on_leg_contact"] = False
    if a.opponent: cfg["frozen_opponent"] = list(a.opponent)
    from koth.action_map import ckpt_spec
    cfg["action"] = ckpt_spec(a.ckpt)
    env = KothEnv(cfg, a.num_envs, seed=0)
    from scripts.eval import load_policy
    policy = load_policy(env, a.ckpt)
    R = cfg["reward"]; steps = int(a.seconds / env.ctrl_dt); lr = env.lrows if env.frozen is not None else slice(None)
    sums = {}; pos = {}; neg = {}; n = 0; bind = 0.0; erased = 0.0; pen_mass = 0.0
    ep_shaped = torch.zeros(env.num_envs, device=env.device); ends = {"won": [], "lost": [], "drawn": [], "bell": []}
    obs = env.reset()
    for _ in range(steps):
        with torch.no_grad(): act = policy(obs)
        obs, rew, done, ex = env.step(act)
        terms = {k: v[lr] for k, v in env.reward_terms.items()}; pre = env._pre_clip[lr]
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + float(v.sum()); pos[k] = pos.get(k, 0.0) + float(v.clamp(min=0).sum()); neg[k] = neg.get(k, 0.0) + float(v.clamp(max=0).sum())
        shaped_pen = sum(v.clamp(max=0) for k, v in terms.items() if k not in TERMINAL)
        n += pre.numel(); bind += float((pre < 0).sum()); erased += float((-pre).clamp(min=0).sum()); pen_mass += float(-shaped_pen.sum())
        ep_shaped += pre.clamp(min=0) if cfg.get("reward_clip", True) else pre
        d = done if done.shape[0] == env.num_envs else done[lr]
        if d.any():
            i = d.nonzero(as_tuple=True)[0]; sh = ep_shaped[i]
            win = terms.get("win", torch.zeros_like(pre))[i] > 0; term = terms.get("termination", torch.zeros_like(pre))[i] < 0
            drew = terms.get("draw", torch.zeros_like(pre))[i] < 0; to = ex["time_outs"][i] if ex["time_outs"].shape[0] == env.num_envs else ex["time_outs"][lr][i]
            for key, m in (("won", win), ("lost", term & ~win), ("drawn", drew), ("bell", to & ~drew & ~win & ~term)):
                ends[key] += sh[m].tolist()
            ep_shaped[i] = 0
    total_pos = sum(v for k, v in pos.items() if k not in TERMINAL) or 1e-9
    rows = {k: dict(per_step=sums[k] / n, pos_share=pos[k] / total_pos if k not in TERMINAL else None, neg_sum_per_step=neg[k] / n) for k in sums}
    mean = lambda x: sum(x) / len(x) if x else None
    out = dict(rung=a.rung, ckpt=a.ckpt, steps=n, clip_binds_share=bind / n, clip_erased_share_of_penalties=erased / max(pen_mass, 1e-9),
               terms=rows, episodes={k: dict(count=len(v), mean_shaped_return=mean(v)) for k, v in ends.items()}, flags=[])
    F = out["flags"]
    if out["clip_binds_share"] > 0.10: F.append(f"clip binds on {out['clip_binds_share']:.0%} of steps, erasing {out['clip_erased_share_of_penalties']:.0%} of the penalty mass")
    for k, r in rows.items():
        if r["pos_share"] is not None and r["pos_share"] > 0.6: F.append(f"dominance: {k} carries {r['pos_share']:.0%} of the positive shaped reward")
    win_r, draw_r, term_r = R.get("win", 0.0), R.get("draw", 0.0), R.get("termination", 0.0)
    ep_len = cfg["episode_length_s"] / env.ctrl_dt
    shaped_step = sum(r["per_step"] for k, r in rows.items() if k not in TERMINAL)
    if env.A >= 2 and win_r:
        dec = out["episodes"]["won"]["mean_shaped_return"]; bell = out["episodes"]["drawn"]["mean_shaped_return"] or out["episodes"]["bell"]["mean_shaped_return"]
        stall = (bell if bell is not None else max(shaped_step, 0) * ep_len) + draw_r; decided = (dec if dec is not None else 0.0) + win_r
        out["stall"] = dict(bell_round_return=stall, won_round_return=decided)
        if stall > decided: F.append(f"stall pays: a round to the bell returns {stall:.1f} vs a win {decided:.1f}")
        for k, r in rows.items():
            if k not in TERMINAL and r["per_step"] > 0 and r["per_step"] * ep_len > win_r: F.append(f"farm: {k} can pay {r['per_step'] * ep_len:.1f} per round, more than a win ({win_r:g})")
    print(f"{a.rung} {os.path.basename(os.path.dirname(a.ckpt)) or a.ckpt}: {n} learner steps")
    print(f"{'term':>20} {'per step':>10} {'+ share':>8}")
    for k, r in sorted(rows.items(), key=lambda kv: -abs(kv[1]["per_step"])):
        share = "" if r["pos_share"] is None else format(r["pos_share"], "8.1%")
        print(f"{k:>20} {r['per_step']:10.5f} {share}")
    print(f"clip binds {out['clip_binds_share']:.1%} of steps; erases {out['clip_erased_share_of_penalties']:.1%} of the shaped penalty mass")
    print("episodes:", {k: (v["count"], None if v["mean_shaped_return"] is None else round(v["mean_shaped_return"], 2)) for k, v in out["episodes"].items()})
    print("FLAGS:", F or "none")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
