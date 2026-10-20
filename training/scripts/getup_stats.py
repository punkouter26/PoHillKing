"""Get-up statistics in CPU MuJoCo, the reference for Unity's get-up gate (ParityBatch.RunGetupStats, GetupProbe.cs).

  uv run python scripts/getup_stats.py --policy getup --drops 200 [--seconds 8]

Each drop uses the training recipe (koth/env.py, rung getup): the robot at the keyframe x/y, pelvis drop_z (0.85 m) above
the floor with a uniformly random orientation, joints at default +- 0.6 rad (clamped to their ranges), zero velocity;
then the get-up brain (103 inputs, zero command, ctrl = default + 0.5 action) runs with PolicyRunner's tick order. A drop
is a success when the robot stands (pelvis up-axis z > 0.9 and height > 0.85 x the standing height, held 0.4 s, the
game's DemoDirector rule) within --count seconds (the game's get-up count, 5 s). Writes training/assets/<robot>/
getup_stats_python.json: success share, median and quartiles of the time to stand."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco, onnxruntime as ort
from koth.obs import GAIT_FREQ_HZ
from koth.build_mjcf import ASSETS, UNITY_MODELS
from koth import cpu_sim

DROP_Z, JOINT_NOISE, STAND_H, STAND_UP, HOLD_S = 0.85, 0.6, 0.85, 0.9, 0.4


def drop(m, d, s, rng, default):
    """The env's drop recipe on robot s (keyframe already reset)."""
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    rq = s["rq"]; d.qpos[rq + 2] = DROP_Z; d.qpos[rq + 3:rq + 7] = q
    jlo, jhi = m.jnt_range[[m.dof_jntid[a] for a in s["dadr"]], 0], m.jnt_range[[m.dof_jntid[a] for a in s["dadr"]], 1]
    d.qpos[s["qadr"]] = np.clip(default + rng.uniform(-JOINT_NOISE, JOINT_NOISE, 29), jlo, jhi)
    d.qvel[:] = 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="getup"); ap.add_argument("--drops", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=8.0); ap.add_argument("--count", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
    sess = ort.InferenceSession(os.path.join(UNITY_MODELS, f"{a.policy}_policy.onnx"), providers=["CPUExecutionProvider"])
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "scene_flat_1p_train.xml")); d = mujoco.MjData(m)
    s = cpu_sim.robot(m, spec["joints"], "a_"); boxes = cpu_sim.pool(m, spec["pool_bodies"])
    default = np.array(spec["default_pose"]); key_z = spec["key_root_z"]; rng = np.random.default_rng(a.seed)
    times = []; stood_ever = 0
    for k in range(a.drops):
        mujoco.mj_resetDataKeyframe(m, d, 0); parked = cpu_sim.park(d, boxes); drop(m, d, s, rng, default)
        mujoco.mj_forward(m, d)
        s["last"] = np.zeros(29, dtype=np.float32); phase = np.float32(0); since = None; t_stand = None
        for tick in range(int(a.seconds / spec["ctrl_dt"])):
            obs = cpu_sim.observe(d, s, default, np.zeros(3), phase)[0].numpy().astype(np.float32)
            act = np.clip(sess.run(None, {"obs": obs[None]})[0][0], -1, 1).astype(np.float32)
            s["ctrl"] = np.clip(default + spec["action_scale"] * act.astype(np.float64), s["lo"], s["hi"]); s["last"] = act
            phase = np.float32(phase + np.float32(2 * np.pi * GAIT_FREQ_HZ * spec["ctrl_dt"]))
            if phase > 2 * np.pi: phase = np.float32(phase - np.float32(2 * np.pi))
            cpu_sim.step(m, d, boxes, parked, [s], spec["decimation"])
            rq = s["rq"]; quat = d.qpos[rq + 3:rq + 7]; up = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2); t = (tick + 1) * spec["ctrl_dt"]
            if up > STAND_UP and d.qpos[rq + 2] > STAND_H * key_z:
                since = t if since is None else since
                if t - since >= HOLD_S and t_stand is None: t_stand = t
            else: since = None
        if t_stand is not None: stood_ever += 1
        if t_stand is not None and t_stand <= a.count: times.append(t_stand)
        if (k + 1) % 25 == 0: print(f"{k + 1} drops: up within {a.count:.0f} s {len(times)}, ever {stood_ever}", flush=True)
    q = lambda f: round(float(np.percentile(times, f)), 2) if times else -1
    res = dict(drops=a.drops, success=round(len(times) / a.drops, 4), stood_within_seconds=round(stood_ever / a.drops, 4), count_s=a.count, seconds=a.seconds,
               median_time_s=q(50), p25_time_s=q(25), p75_time_s=q(75))
    json.dump(res, open(os.path.join(ASSETS, "getup_stats_python.json"), "w"), indent=1); print(json.dumps(res))


if __name__ == "__main__":
    main()
