"""Interactive MuJoCo viewer for a training scene. No policy: robots hold the keyframe stance (passive PD).
  uv run python scripts/view.py [scene_koth_2p_train.xml] [--no-boxes]
Every 4 s a pool box is fired at a robot, the same way training and Unity do it (write qpos/qvel of the free joint).
Viewer keys: space = pause, backspace = reset, double-click + ctrl-drag = push a body."""
import os, sys, time
import mujoco, mujoco.viewer, numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "g1")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
scene = os.path.join(ASSETS, args[0] if args else "scene_koth_2p_train.xml")
boxes = "--no-boxes" not in sys.argv

m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
J = mujoco.mjtObj.mjOBJ_JOINT
roots = [m.jnt_qposadr[j] for j in (mujoco.mj_name2id(m, J, p + "floating_base_joint") for p in ("a_", "b_")) if j >= 0]
pool = [(m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in (mujoco.mj_name2id(m, J, f"box{i}_free") for i in range(8)) if j >= 0]
rng = np.random.default_rng(0)
shot = 0

with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.distance, v.cam.elevation, v.cam.azimuth, v.cam.lookat[:] = 6.0, -15, 90, (0, 0, 0.6)
    next_fire = 3.0
    while v.is_running():
        t0 = time.perf_counter()
        if boxes and pool and d.time >= next_fire:
            target = d.qpos[roots[shot % len(roots)]:][:3] + np.array([0, 0, 0.1])
            ang = rng.uniform(0, 2 * np.pi)
            start = target + np.array([3 * np.cos(ang), 3 * np.sin(ang), 0.3])
            vel = (target - start) / np.linalg.norm(target - start) * rng.uniform(3, 8)
            q, dof = pool[shot % len(pool)]
            d.qpos[q:q + 7] = [*start, 1, 0, 0, 0]; d.qvel[dof:dof + 6] = [*vel, 0, 0, 0]
            shot += 1; next_fire = d.time + 4.0
        for _ in range(8):                      # 8 x 2 ms = 16 ms per frame, ~real time at 60 fps
            mujoco.mj_step(m, d)
        if d.time < 0.02:                       # viewer reset (backspace) -> restore the stance
            mujoco.mj_resetDataKeyframe(m, d, 0); next_fire = 3.0
        v.sync()
        time.sleep(max(0, 0.016 - (time.perf_counter() - t0)))
