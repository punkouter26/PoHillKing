"""How a fighter behaves while it is within reach of the opponent (0.9 m): jitter, arm use, hands on the opponent.
  uv run python scripts/contact_style.py --ckpt runs/league/careful1.pt --opponent runs/league/careful1.pt [--seconds 15]
Deterministic policy, r4league evaluation settings, robot a_ only. Prints one JSON line.
  action_change   mean over joints of |change per control step| in what drives the joint targets: the action, or the
                  low-passed action in the shove style (0 = perfectly smooth, 2 = bang-bang)
  arm_off_rest    mean |joint angle - standing pose| over waist/shoulder/elbow/wrist joints, radians
  hand_reach      how far the hands are from the pelvis toward the opponent, metres, mean of both (about 0 hanging, 0.35 pushing)
--no-noise switches the observation noise off, as in Unity."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from koth.env import KothEnv, default_cfg
from scripts.eval import load_policy

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--opponent", required=True, nargs="+")
ap.add_argument("--num-envs", type=int, default=256); ap.add_argument("--seconds", type=float, default=15.0)
ap.add_argument("--no-noise", action="store_true")
ap.add_argument("--shove", action="store_true", help="the tested policy fights in the shove style (arm pose + action smoothing)")
a = ap.parse_args()
cfg = default_cfg("r4league"); cfg["max_radius"] = 1.7
cfg["frozen_opponent"] = list(a.opponent)
cfg["shove"] = a.shove
if a.no_noise: cfg["noise"] = {k: 0.0 for k in cfg["noise"]}
cfg["push"] = dict(interval_s=[1e6, 1e6], vel=[0.0, 0.0]); cfg["projectile"]["enabled"] = False
env = KothEnv(cfg, a.num_envs, seed=0); policy = load_policy(env, a.ckpt)
torch.manual_seed(1000); obs = env.reset()
n = change = arm = hands = 0.0; total = 0
for _ in range(int(a.seconds / env.ctrl_dt)):
    prev = (env.act_f if a.shove else env.last_action)[0::2].clone()
    with torch.no_grad(): act = policy(obs)
    obs, _, done, extras = env.step(act)
    near = (extras["sep"][0::2] < 0.9) & ~done                      # rows that reset this step hold a new episode's state
    jpos, _ = env._joints(); dev = (jpos - env.default_pose)[0::2][:, env.upper_idx].abs().mean(1)
    pos = env._root()[0]; to = env._opp(pos)[:, :2] - pos[:, :2]; u = to / to.norm(dim=1).clamp(min=0.1)[:, None]
    touch = (((env._flat(env.xpos[:, env.hand_body])[:, :, :2] - pos[:, None, :2]) * u[:, None]).sum(2).mean(1))[0::2]
    k = near.float()
    n += k.sum().item(); total += k.numel()
    change += (((env.act_f if a.shove else env.action)[0::2] - prev).abs().mean(1) * k).sum().item(); arm += (dev * k).sum().item(); hands += (touch.float() * k).sum().item()
n = max(n, 1.0)
print(json.dumps(dict(ckpt=os.path.basename(a.ckpt), in_reach_share=round(n / total, 3), action_change=round(change / n, 3),
                      arm_off_rest=round(arm / n, 3), hand_reach=round(hands / n, 3), obs_noise=not a.no_noise)))
