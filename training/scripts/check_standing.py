"""A.8: hold the default pose with no policy in mujoco_warp for T seconds across N worlds.
Prints the pelvis-height trace used by Unity's zero-brain parity test (B.10)."""
import json, os, sys, time
import mujoco, numpy as np, warp as wp, mujoco_warp as mjw

wp.init()
scene = sys.argv[1] if len(sys.argv) > 1 else "assets/g1/scene_koth_2p.xml"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
T = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
mjm = mujoco.MjModel.from_xml_path(scene)
mjd = mujoco.MjData(mjm)
mujoco.mj_resetDataKeyframe(mjm, mjd, 0); mujoco.mj_forward(mjm, mjd)
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=N, nconmax=64, njmax=400)   # per-world budgets
prefixes = [p for p in ("a_", "b_") if mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, p + "pelvis") >= 0]
roots = {p: mjm.jnt_qposadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, p + "floating_base_joint")] for p in prefixes}
with wp.ScopedCapture() as cap: mjw.step(m, d)
steps = int(round(T / mjm.opt.timestep)); marks = {int(round(t / mjm.opt.timestep)) for t in (0.5, 1, 3, 5, 10) if t <= T}
trace = {}
t0 = time.perf_counter()
for i in range(1, steps + 1):
    wp.capture_launch(cap.graph)
    if i in marks:
        q = d.qpos.numpy()
        trace[round(i * mjm.opt.timestep, 3)] = {p: [float(q[:, r + 2].mean()), float(q[:, r + 2].std())] for p, r in roots.items()}
wp.synchronize()
q = d.qpos.numpy()
print(f"{scene}: {N} worlds x {steps} steps in {time.perf_counter() - t0:.1f}s  ({N * steps / (time.perf_counter() - t0):,.0f} env-steps/s)")
ok = True
for t, v in trace.items():
    print(f"t={t:>5}s  " + "  ".join(f"{p}pelvis z={mu:.4f}±{sd:.4f}" for p, (mu, sd) in v.items()))
for p, r in roots.items():
    z = q[:, r + 2]
    up = 1 - 2 * (q[:, r + 4] ** 2 + q[:, r + 5] ** 2)   # world z of body z-axis from quaternion (w,x,y,z)
    print(f"{p}: final z min {z.min():.3f}  upright(min z-axis·up) {up.min():.3f}  nan {bool(np.isnan(q).any())}")
    ok &= bool(z.min() > 0.6 and up.min() > 0.9 and not np.isnan(q).any())
json.dump({"scene": os.path.basename(scene), "trace": trace}, open("assets/g1/hold_trace_" + os.path.basename(scene).replace(".xml", ".json"), "w"), indent=1)
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
