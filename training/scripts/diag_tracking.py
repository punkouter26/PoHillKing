"""Tracking diagnostics with disturbances and noise off: per fixed command, mean and std of the achieved body
velocity. uv run python scripts/diag_tracking.py --ckpt runs/r1/r1_a/model_1000.pt"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rsl_rl.runners import OnPolicyRunner
from koth.env import KothEnv, default_cfg
from scripts.train import train_cfg

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); a = ap.parse_args()
cfg = default_cfg("r1")
cfg["push"] = dict(interval_s=[1e6, 1e6], vel=[0, 0]); cfg["projectile"]["enabled"] = False
cfg["noise"] = {k: 0.0 for k in cfg["noise"]}; cfg["command"]["resample_s"] = 1e6
cmds = [(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (-0.5, 0, 0), (0, 0.4, 0), (0, 0, 0.8), (0, 0, -0.8), (0.6, 0, 0.6)]
env = KothEnv(cfg, 64 * len(cmds))
runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cuda"); runner.load(a.ckpt)
policy = runner.alg.get_policy(); policy.eval()
obs = env.reset()
C = torch.tensor(cmds, device=env.device, dtype=torch.float32).repeat_interleave(64, 0)
env.command[:] = C; obs = env.get_observations()
vel = []; falls = torch.zeros(len(cmds), device=env.device)
for t in range(500):
    with torch.no_grad(): act = policy(obs)
    obs, _, _, ex = env.step(act); env.command[:] = C; obs = env.get_observations()
    falls += ex["fallen"].view(len(cmds), 64).sum(1)
    if t >= 150: vel.append(torch.cat([obs["critic"][:, 0:2], obs["critic"][:, 5:6]], 1))
V = torch.stack(vel)                                   # (T, M, 3)
mean = V.mean(0).view(len(cmds), 64, 3).mean(1); osc = V.std(0).view(len(cmds), 64, 3).mean(1)
print("command (vx vy wz)      achieved mean (vx vy wz)      oscillation std (vx vy wz)    falls")
for i, c in enumerate(cmds):
    print(f"{c[0]:5.1f} {c[1]:4.1f} {c[2]:5.1f}       {mean[i,0]:6.2f} {mean[i,1]:6.2f} {mean[i,2]:6.2f}          {osc[i,0]:5.2f} {osc[i,1]:5.2f} {osc[i,2]:5.2f}        {int(falls[i])}")
