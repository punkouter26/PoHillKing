"""CPU MuJoCo helpers for the scripts that run policies outside Warp (duel_stats, getup_stats, play, record_reference,
view, export_mlp, onnx_to_ckpt).

The tick is Unity's PolicyRunner order: every robot observes the same pre-step state, infers, writes ctrl; then
`decimation` physics steps, with the parked pool boxes re-pinned (qpos = park pose, qvel = 0) before each one.
observe() builds the observation with koth/obs.py in float64 from mjData; callers keep their own precision for the
inputs they own (last action, command, phase, default pose) and their own rounding of the result."""
from __future__ import annotations
import json, os
import numpy as np, torch, mujoco
from koth.obs import build_obs, build_combat, GAIT_FREQ_HZ

J, ACT = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR


def joint_map(body: str | None = None) -> dict:
    """assets/<body>/joint_map.json; None = the KOTH_ROBOT body (koth.build_mjcf.ASSETS)."""
    from koth.build_mjcf import ASSETS
    return json.load(open(os.path.join(os.path.dirname(ASSETS), body, "joint_map.json") if body else os.path.join(ASSETS, "joint_map.json")))


def session(onnx_path: str):
    """onnxruntime CPU session (the brains are batch-1 MLPs with one input, "obs")."""
    import onnxruntime as ort
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def infer(sess, obs: np.ndarray) -> np.ndarray:
    """Raw network output for one float32 observation."""
    return sess.run(None, {"obs": obs[None]})[0][0]


def onnx_rows(sess, x: np.ndarray) -> np.ndarray:
    """The session's output for each row of x, one batch-1 call per row (the exported graphs have batch 1)."""
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])


def check_inputs(dim: int, n: int = 1000) -> np.ndarray:
    """The fixed random inputs (seed 0, std 2, float32) the export checks compare torch / numpy against ONNX on."""
    return np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32) * 2


def advance_phase(phase, ctrl_dt: float) -> np.float32:
    """PolicyRunner's gait phase step in float32, wrapped once past 2 pi."""
    p = np.float32(phase + np.float32(2 * np.pi * GAIT_FREQ_HZ * ctrl_dt))
    return np.float32(p - np.float32(2 * np.pi)) if p > 2 * np.pi else p


def default_ctrl(s: dict, default, scale: float, act) -> np.ndarray:
    """ctrl = default + scale * action in float64, clamped to robot s's ctrl range (the plain action path)."""
    return np.clip(np.asarray(default, dtype=np.float64) + scale * np.asarray(act, dtype=np.float64), s["lo"], s["hi"])


def upright(quat) -> float:
    """World z of the body's z axis from a (w, x, y, z) quaternion."""
    return 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)


def T(x) -> torch.Tensor:
    """(1, n) float64 tensor."""
    return torch.tensor(np.asarray(x, dtype=np.float64))[None]


def robot(m: mujoco.MjModel, joints: list[str], prefix: str) -> dict:
    """Index table of one robot: joint qpos/dof addresses and actuator ids in canonical order, root addresses,
    ctrl range."""
    jid = [mujoco.mj_name2id(m, J, prefix + j) for j in joints]; root = mujoco.mj_name2id(m, J, prefix + "floating_base_joint")
    aid = np.array([mujoco.mj_name2id(m, ACT, prefix + j) for j in joints])
    return dict(qadr=np.array([m.jnt_qposadr[j] for j in jid]), dadr=np.array([m.jnt_dofadr[j] for j in jid]), aid=aid,
                rq=m.jnt_qposadr[root], rd=m.jnt_dofadr[root], lo=m.actuator_ctrlrange[aid, 0], hi=m.actuator_ctrlrange[aid, 1])


def robots(m: mujoco.MjModel, joints: list[str], prefixes) -> list[dict]:
    return [robot(m, joints, p) for p in prefixes]


def pool(m: mujoco.MjModel, names=None) -> list[tuple[int, int]]:
    """(qpos address, dof address) of each pool box's free joint. names = joint_map "pool_bodies"; None = every
    box<i> the model has."""
    if names is None:
        names = [f"box{i}" for i in range(64) if mujoco.mj_name2id(m, J, f"box{i}_free") >= 0]
    return [(m.jnt_qposadr[mujoco.mj_name2id(m, J, b + "_free")], m.jnt_dofadr[mujoco.mj_name2id(m, J, b + "_free")]) for b in names]


def park(d: mujoco.MjData, boxes) -> list[np.ndarray]:
    """The boxes' current qpos (call right after the keyframe reset: the park pose)."""
    return [d.qpos[q:q + 7].copy() for q, _ in boxes]


def pin(d: mujoco.MjData, boxes, parked) -> None:
    for (q, dof), pk in zip(boxes, parked):
        d.qpos[q:q + 7] = pk; d.qvel[dof:dof + 6] = 0


def step(m: mujoco.MjModel, d: mujoco.MjData, boxes, parked, rs, decimation: int) -> None:
    """One control tick of physics: re-pin the pool, write each robot's s["ctrl"], mj_step; `decimation` times."""
    for _ in range(decimation):
        pin(d, boxes, parked)
        for s in rs: d.ctrl[s["aid"]] = s["ctrl"]
        mujoco.mj_step(m, d)


def observe(d: mujoco.MjData, s: dict, default, cmd, phase, other: dict | None = None) -> torch.Tensor:
    """(1, 103) float64 observation of robot s (koth.obs.build_obs), plus the 9 combat inputs against `other` when
    given. Uses s["last"] as the last action. default/cmd/last/phase are converted to float64 exactly as given."""
    rq, rd = s["rq"], s["rd"]; quat = d.qpos[rq + 3:rq + 7]
    obs = build_obs(T(quat), T(d.qvel[rd:rd + 3]), T(d.qvel[rd + 3:rd + 6]), T(d.qpos[s["qadr"]]), T(d.qvel[s["dadr"]]),
                    torch.tensor(np.asarray(default, dtype=np.float64)), T(s["last"]), T(cmd), torch.tensor([float(phase)], dtype=torch.float64))
    if other is not None:
        oq, od = other["rq"], other["rd"]
        obs = torch.cat([obs, build_combat(T(d.qpos[rq:rq + 3]), T(quat), T(d.qvel[rd:rd + 3]), T(d.qpos[oq:oq + 3]),
                                           T(d.qpos[oq + 3:oq + 7]), T(d.qvel[od:od + 3]))], 1)
    return obs
