"""CPU MuJoCo helpers for the scripts that run policies outside Warp (duel_stats, play, record_reference, view).

The tick is Unity's PolicyRunner order: every robot observes the same pre-step state, infers, writes ctrl; then
`decimation` physics steps, with the parked pool boxes re-pinned (qpos = park pose, qvel = 0) before each one.
observe() builds the observation with koth/obs.py in float64 from mjData; callers keep their own precision for the
inputs they own (last action, command, phase, default pose) and their own rounding of the result."""
from __future__ import annotations
import numpy as np, torch, mujoco
from koth.obs import build_obs, build_combat

J, ACT = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR


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
