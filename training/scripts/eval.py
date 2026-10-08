"""Pass-bar evaluation for a checkpoint. Deterministic policy, domain randomization on, obs noise on.
  uv run python scripts/eval.py --rung r0 --ckpt runs/r0/r0_b/model_1499.pt [--seeds 10 --num-envs 256]

R0 bar: stand 20 s under 2 m/s pelvis kicks every 5 s and 2 kg boxes at 6 m/s. Pass when, in every seed,
        at least 90 % of impacts are survived (a fall within the episode counts against the preceding impacts).
R1 bar: R0 disturbances + random velocity commands; tracking error < 0.15 m/s and < 0.2 rad/s, falls as R0.
Prints one JSON line per seed plus a verdict, and exits 0 on pass, 1 on fail."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rsl_rl.runners import OnPolicyRunner
from koth.env import KothEnv, default_cfg
from scripts.train import train_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    cfg = default_cfg(a.rung)
    cfg["push"] = dict(interval_s=[5.0, 5.0], vel=[2.0, 2.0])
    cfg["projectile"].update(speed=[6.0, 6.0])
    env = KothEnv(cfg, a.num_envs, seed=0)
    runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cuda")
    runner.load(a.ckpt)
    policy = runner.alg.get_policy(); policy.eval()
    steps = int(a.seconds / env.ctrl_dt); ok = True; rows = []
    for seed in range(a.seeds):
        torch.manual_seed(1000 + seed)
        obs = env.reset()
        falls = 0; impacts = 0; lin_err = []; ang_err = []
        ever_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for _ in range(steps):
            push_due = (env.push_timer - env.ctrl_dt <= 0).sum(); box_due = (env.proj_timer - env.ctrl_dt <= 0).sum()
            with torch.no_grad():
                act = policy(obs)
            obs, _, _, extras = env.step(act)
            impacts += int(push_due + box_due); falls += int(extras["fallen"].sum()); ever_fell |= extras["fallen"]
            o = obs["critic"]
            lin_err.append((env.command[:, :2] - o[:, 0:2]).norm(dim=1).mean().item()); ang_err.append((env.command[:, 2] - o[:, 5]).abs().mean().item())
        surv = 1 - falls / max(1, impacts)
        row = dict(seed=seed, impacts=impacts, falls=falls, impact_survival=round(surv, 4), envs_never_fell=round(1 - ever_fell.float().mean().item(), 4),
                   lin_vel_err=round(sum(lin_err) / len(lin_err), 4), ang_vel_err=round(sum(ang_err) / len(ang_err), 4))
        passed = surv >= 0.9 and (a.rung == "r0" or (row["lin_vel_err"] < 0.15 and row["ang_vel_err"] < 0.2))
        row["pass"] = passed; ok &= passed; rows.append(row); print(json.dumps(row), flush=True)
    print("VERDICT", a.rung, "PASS" if ok else "FAIL", f"({sum(r['pass'] for r in rows)}/{len(rows)} seeds)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
