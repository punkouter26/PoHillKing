"""Outcome statistics for attacker-vs-attacker rounds in CPU MuJoCo, with the same rules and spawn recipe as Unity's
DemoDirector (both on the summit at radius 0.5-1.2 m on opposite bearings, random yaw; out = radius > 1.7 m,
fallen, or the 25 s bell). Used for the statistical Unity parity gate of the combat rungs.

  uv run python scripts/duel_stats.py --policy attacker --rounds 200
Writes training/assets/g1/duel_stats_python.json."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, mujoco, onnxruntime as ort
from koth.obs import build_obs, build_combat, goal_command, GAIT_FREQ_HZ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ASSETS = os.path.join(ROOT, "assets", "g1")
MODELS = os.path.normpath(os.path.join(ROOT, "..", "Assets", "PoKingHill", "Models"))
ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="attacker"); ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--seconds", type=float, default=25.0); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
sess = ort.InferenceSession(os.path.join(MODELS, f"{a.policy}_policy.onnx"), providers=["CPUExecutionProvider"])
m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "scene_koth_2p_train.xml")); d = mujoco.MjData(m)
J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
R = []
for p in ("a_", "b_"):
    jid = [mujoco.mj_name2id(m, J, p + j) for j in spec["joints"]]; root = mujoco.mj_name2id(m, J, p + "floating_base_joint")
    aid = np.array([mujoco.mj_name2id(m, A, p + j) for j in spec["joints"]])
    R.append(dict(qadr=np.array([m.jnt_qposadr[j] for j in jid]), dadr=np.array([m.jnt_dofadr[j] for j in jid]), aid=aid,
                  rq=m.jnt_qposadr[root], rd=m.jnt_dofadr[root], lo=m.actuator_ctrlrange[aid, 0], hi=m.actuator_ctrlrange[aid, 1]))
pool = [(m.jnt_qposadr[mujoco.mj_name2id(m, J, b + "_free")], m.jnt_dofadr[mujoco.mj_name2id(m, J, b + "_free")]) for b in spec["pool_bodies"]]
default = np.array(spec["default_pose"]); rng = np.random.default_rng(a.seed)
T = lambda x: torch.tensor(np.asarray(x, dtype=np.float64))[None]
wins = [0, 0]; ties = 0; times = []
for rnd in range(a.rounds):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    bearing = rng.uniform(0, 2 * np.pi)
    for i, s in enumerate(R):
        r = rng.uniform(0.5, 1.2); b = bearing + i * np.pi; yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[s["rq"]:s["rq"] + 2] = [r * np.cos(b), r * np.sin(b)]; d.qpos[s["rq"] + 3:s["rq"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        s["last"] = np.zeros(29, dtype=np.float32); s["phase"] = np.float32(0)
    park = [d.qpos[q:q + 7].copy() for q, _ in pool]
    result = None
    for tick in range(int(a.seconds / spec["ctrl_dt"])):
        for i, s in enumerate(R):
            o = R[1 - i]; rq, rd = s["rq"], s["rd"]; quat = d.qpos[rq + 3:rq + 7]
            cmd = goal_command(T(quat), T([d.qpos[o["rq"]] - d.qpos[rq], d.qpos[o["rq"] + 1] - d.qpos[rq + 1]]), 0.0, 1.0)[0].numpy()
            obs = torch.cat([build_obs(T(quat), T(d.qvel[rd:rd + 3]), T(d.qvel[rd + 3:rd + 6]), T(d.qpos[s["qadr"]]), T(d.qvel[s["dadr"]]),
                                       torch.tensor(default), T(s["last"]), T(cmd), torch.tensor([float(s["phase"])], dtype=torch.float64)),
                             build_combat(T(d.qpos[rq:rq + 3]), T(quat), T(d.qvel[rd:rd + 3]), T(d.qpos[o["rq"]:o["rq"] + 3]),
                                          T(d.qpos[o["rq"] + 3:o["rq"] + 7]), T(d.qvel[o["rd"]:o["rd"] + 3]))], 1)[0].numpy().astype(np.float32)
            act = np.clip(sess.run(None, {"obs": obs[None]})[0][0], -1, 1).astype(np.float32)
            s["ctrl"] = np.clip(default + spec["action_scale"] * act.astype(np.float64), s["lo"], s["hi"]); s["last"] = act
            s["phase"] = np.float32(s["phase"] + np.float32(2 * np.pi * GAIT_FREQ_HZ * spec["ctrl_dt"]))
            if s["phase"] > 2 * np.pi: s["phase"] = np.float32(s["phase"] - np.float32(2 * np.pi))
        for _ in range(spec["decimation"]):
            for (q, dof), pk in zip(pool, park):
                d.qpos[q:q + 7] = pk; d.qvel[dof:dof + 6] = 0
            for s in R: d.ctrl[s["aid"]] = s["ctrl"]
            mujoco.mj_step(m, d)
        out = []
        for s in R:
            p = d.qpos[s["rq"]:s["rq"] + 3]; q = d.qpos[s["rq"] + 3:s["rq"] + 7]; rad = np.hypot(p[0], p[1])
            out.append(rad > 1.7 or 1 - 2 * (q[1] ** 2 + q[2] ** 2) < 0.3 or p[2] + (1 / 9) * max(rad - 1.5, 0) ** 2 < 0.3)
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
