"""Watch trained policies in the MuJoCo viewer (CPU MuJoCo, same tick order as Unity).

  uv run python scripts/play.py walk  --ckpt runs/r1/r1_b/model_final.pt             # flat ground, cycling commands
  uv run python scripts/play.py duel  --ckpt runs/r4att/r4att_c/model_200.pt --opponent runs/r1/r1_b/model_final.pt
  uv run python scripts/play.py approach --ckpt runs/r1/r1_b/model_final.pt          # two robots walk up to each other

With no --ckpt the newest checkpoint of the matching run folder is used. Rounds restart automatically.
Viewer keys: space = pause, double-click a body then ctrl + right-drag = shove it."""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, mujoco, mujoco.viewer
from koth.obs import goal_command, GAIT_FREQ_HZ, OBS_DIM
from koth.build_mjcf import ASSETS
from koth.rules import is_out
from koth import actor, cpu_sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Policy:
    """Checkpoint actor in float64 on the CPU (koth.actor), clamped to the action range."""
    def __init__(self, ckpt):
        self.net = actor.load(ckpt, dtype=torch.float64); self.obs_dim = self.net.obs_dim

    @torch.no_grad()
    def __call__(self, obs):
        return self.net(obs[:, :self.obs_dim]).clamp(-1, 1)[0].numpy()


def newest(pattern):
    files = glob.glob(os.path.join(ROOT, pattern))
    return max(files, key=os.path.getmtime) if files else None


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["walk", "approach", "duel", "mirror"]); ap.add_argument("--ckpt", default=None); ap.add_argument("--opponent", default=None)
a = ap.parse_args()
spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
if a.mode == "walk":
    scene, prefixes, goals = "scene_flat_1p_train.xml", ["a_"], ["cmd"]
    ckpts = [a.ckpt or newest("runs/r1/*/model_*.pt")]
elif a.mode == "approach":
    scene, prefixes, goals = "scene_koth_2p_train.xml", ["a_", "b_"], ["opponent", "opponent"]
    ckpts = [a.ckpt or newest("runs/r1/*/model_*.pt")] * 2
elif a.mode == "mirror":                      # two attackers, both start on the summit
    scene, prefixes, goals = "scene_koth_2p_train.xml", ["a_", "b_"], ["attack", "attack"]
    ckpts = [a.ckpt or newest("runs/league/gen*.pt"), a.opponent or newest("runs/league/gen*.pt")]
else:
    scene, prefixes, goals = "scene_koth_2p_train.xml", ["a_", "b_"], ["attack", "stand"]
    ckpts = [a.ckpt or newest("runs/r4att/*/model_*.pt"), a.opponent or newest("runs/r1/*/model_*.pt")]
print("policies:", [os.path.relpath(c, ROOT) for c in ckpts])
policies = [Policy(c) for c in ckpts]

m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, scene)); d = mujoco.MjData(m)
R = cpu_sim.robots(m, spec["joints"], prefixes)
pool = cpu_sim.pool(m, spec["pool_bodies"])
default = np.array(spec["default_pose"]); rng = np.random.default_rng()
T = cpu_sim.T
CMDS = [(0, 0, 0), (0.8, 0, 0), (0.5, 0, 0.8), (0, 0.4, 0), (-0.5, 0, 0), (0.6, 0, -0.8), (0, 0, 0)]
score = {"a": 0, "b": 0, "draw": 0}


def reset():
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if a.mode != "walk":
        if a.mode == "duel": rs = [rng.uniform(0.0, 0.5), rng.uniform(1.2, 1.35)]
        else: rs = [rng.uniform(0.5, 1.2), rng.uniform(0.5, 1.2)]
        ang = rng.uniform(0, 2 * np.pi)
        for i, s in enumerate(R):
            b = ang + i * np.pi; yaw = rng.uniform(-np.pi, np.pi)
            d.qpos[s["rq"]:s["rq"] + 2] = [rs[i] * np.cos(b), rs[i] * np.sin(b)]
            d.qpos[s["rq"] + 3:s["rq"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    for s in R: s["last"] = np.zeros(29); s["phase"] = 0.0; s["ctrl"] = default.copy()
    return cpu_sim.park(d, pool)


park = reset(); t_round = 0.0
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.distance, v.cam.elevation, v.cam.azimuth, v.cam.lookat[:] = (5.0, -20, 90, (0, 0, 0.6))
    while v.is_running():
        t0 = time.perf_counter()
        for i, s in enumerate(R):
            rq = s["rq"]; quat = d.qpos[rq + 3:rq + 7]
            if goals[i] == "cmd": cmd = np.array(CMDS[int(d.time // 4) % len(CMDS)], dtype=np.float64)
            elif goals[i] == "stand": cmd = np.zeros(3)
            else:
                o = R[1 - i]["rq"]; gv = [d.qpos[o] - d.qpos[rq], d.qpos[o + 1] - d.qpos[rq + 1]]
                stop, vmax = (0.0, 1.0) if goals[i] == "attack" else (0.5, 0.8)
                cmd = goal_command(T(quat), T(gv), stop, vmax)[0].numpy()
            obs = cpu_sim.observe(d, s, default, cmd, s["phase"], other=R[1 - i] if policies[i].obs_dim > OBS_DIM else None)
            act = policies[i](obs)
            s["ctrl"] = np.clip(default + spec["action_scale"] * act, s["lo"], s["hi"]); s["last"] = act
            s["phase"] = (s["phase"] + 2 * np.pi * GAIT_FREQ_HZ * spec["ctrl_dt"]) % (2 * np.pi)
        cpu_sim.step(m, d, pool, park, R, spec["decimation"])
        t_round += spec["ctrl_dt"]
        # round end: someone fell or left the summit, or the bell
        out = [is_out(d.qpos[s["rq"]:s["rq"] + 3], d.qpos[s["rq"] + 3:s["rq"] + 7], arena=a.mode != "walk") for s in R]
        limit = 28.0 if a.mode == "walk" else 15.0
        if any(out) or t_round > limit:
            if a.mode in ("duel", "mirror"):
                k = "b" if out[0] and not out[1] else "a" if out[1] else "draw"; score[k] += 1
                print(f"round over after {t_round:4.1f}s: {'robot A wins' if k == 'a' else 'robot B wins' if k == 'b' else 'draw'}   score A {score['a']} / B {score['b']} / draws {score['draw']}", flush=True)
            v.sync(); time.sleep(0.8)
            park = reset(); t_round = 0.0
        v.sync()
        time.sleep(max(0, spec["ctrl_dt"] - (time.perf_counter() - t0)))
