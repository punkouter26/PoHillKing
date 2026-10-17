"""Outcome statistics for attacker-vs-attacker rounds in CPU MuJoCo, with the same rules and spawn recipe as Unity's
DemoDirector (both on the summit at radius 0.5-1.2 m on opposite bearings, random yaw; out = koth.rules.is_out:
radius > 1.7 m, fallen, or the 25 s bell). Used for the statistical Unity parity gate of the combat rungs.

  uv run python scripts/duel_stats.py --policy attacker --rounds 200
Writes training/assets/<robot>/duel_stats_python.json (policies from Assets/PoKingHill/Models[/<robot>])."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, mujoco, onnxruntime as ort
from koth.obs import goal_command, GAIT_FREQ_HZ, SHOVE_POSE, SHOVE_FAR, SHOVE_NEAR, SHOVE_SMOOTH
from koth.build_mjcf import ASSETS, UNITY_MODELS
from koth.rules import is_out
from koth import cpu_sim

ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="attacker"); ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--seconds", type=float, default=25.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shove", action="store_true", help="both robots use the shove style (arm pose + action smoothing), as in koth/env.py"); a = ap.parse_args()
spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
sess = ort.InferenceSession(os.path.join(UNITY_MODELS, f"{a.policy}_policy.onnx"), providers=["CPUExecutionProvider"])
m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "scene_koth_2p_train.xml")); d = mujoco.MjData(m)
R = cpu_sim.robots(m, spec["joints"], ("a_", "b_"))
pool = cpu_sim.pool(m, spec["pool_bodies"])
default = np.array(spec["default_pose"])
shove_offset = np.array([next((v - default[i] for k, v in SHOVE_POSE.items() if k in j), 0.0) for i, j in enumerate(spec["joints"])])
rng = np.random.default_rng(a.seed)
T = cpu_sim.T
wins = [0, 0]; ties = 0; times = []
for rnd in range(a.rounds):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    bearing = rng.uniform(0, 2 * np.pi)
    for i, s in enumerate(R):
        r = rng.uniform(0.5, 1.2); b = bearing + i * np.pi; yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[s["rq"]:s["rq"] + 2] = [r * np.cos(b), r * np.sin(b)]; d.qpos[s["rq"] + 3:s["rq"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        s["last"] = np.zeros(29, dtype=np.float32); s["f"] = np.zeros(29); s["phase"] = np.float32(0)
    park = cpu_sim.park(d, pool)
    result = None
    for tick in range(int(a.seconds / spec["ctrl_dt"])):
        for i, s in enumerate(R):
            o = R[1 - i]; rq = s["rq"]; quat = d.qpos[rq + 3:rq + 7]
            cmd = goal_command(T(quat), T([d.qpos[o["rq"]] - d.qpos[rq], d.qpos[o["rq"] + 1] - d.qpos[rq + 1]]), 0.0, 1.0)[0].numpy()
            obs = cpu_sim.observe(d, s, default, cmd, s["phase"], other=o)[0].numpy().astype(np.float32)
            act = np.clip(sess.run(None, {"obs": obs[None]})[0][0], -1, 1).astype(np.float32)
            f = act.astype(np.float64); off = 0.0
            if a.shove:
                f = s["f"] = SHOVE_SMOOTH * s["f"] + (1 - SHOVE_SMOOTH) * f
                dist = np.hypot(d.qpos[o["rq"]] - d.qpos[rq], d.qpos[o["rq"] + 1] - d.qpos[rq + 1])
                off = np.clip((SHOVE_FAR - dist) / (SHOVE_FAR - SHOVE_NEAR), 0, 1) * shove_offset
            s["ctrl"] = np.clip(default + off + spec["action_scale"] * f, s["lo"], s["hi"]); s["last"] = act
            s["phase"] = np.float32(s["phase"] + np.float32(2 * np.pi * GAIT_FREQ_HZ * spec["ctrl_dt"]))
            if s["phase"] > 2 * np.pi: s["phase"] = np.float32(s["phase"] - np.float32(2 * np.pi))
        cpu_sim.step(m, d, pool, park, R, spec["decimation"])
        out = [is_out(d.qpos[s["rq"]:s["rq"] + 3], d.qpos[s["rq"] + 3:s["rq"] + 7]) for s in R]
        if any(out):
            if out[0] != out[1]: result = 1 if out[0] else 0; wins[result] += 1; times.append((tick + 1) * spec["ctrl_dt"])
            else: ties += 1
            break
    else:
        ties += 1
    if (rnd + 1) % 25 == 0: print(f"{rnd + 1} rounds: A {wins[0]}  B {wins[1]}  ties {ties}", flush=True)
res = dict(rounds=a.rounds, wins_a=wins[0], wins_b=wins[1], ties=ties, decided=round((wins[0] + wins[1]) / a.rounds, 4),
           median_time_s=round(float(np.median(times)), 2) if times else -1, mean_time_s=round(float(np.mean(times)), 2) if times else -1,
           p25_time_s=round(float(np.percentile(times, 25)), 2) if times else -1, p75_time_s=round(float(np.percentile(times, 75)), 2) if times else -1)
json.dump(res, open(os.path.join(ASSETS, "duel_stats_python.json"), "w"), indent=1); print(json.dumps(res))
