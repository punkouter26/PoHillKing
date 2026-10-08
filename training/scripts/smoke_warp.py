"""0.3 smoke test: step N worlds of a G1 scene in mujoco_warp on GPU, with and without CUDA graph capture."""
import time, sys
import mujoco, warp as wp, mujoco_warp as mjw, numpy as np
wp.init()
xml = sys.argv[1] if len(sys.argv) > 1 else "assets/g1/scene.xml"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
mjm = mujoco.MjModel.from_xml_path(xml)
mjd = mujoco.MjData(mjm)
if mjm.nkey: mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
mujoco.mj_forward(mjm, mjd)
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=N)
print(f"{xml}: device {wp.get_device()} nworld {N} nq {mjm.nq} nu {mjm.nu} dt {mjm.opt.timestep} "
      f"integrator {mjm.opt.integrator} iters {mjm.opt.iterations} ls {mjm.opt.ls_iterations} ngeom {mjm.ngeom}")
for _ in range(3): mjw.step(m, d)
wp.synchronize()
S = 50
t = time.perf_counter()
for _ in range(S): mjw.step(m, d)
wp.synchronize(); el = time.perf_counter() - t
print(f"eager : {N*S/el:>12,.0f} env-steps/s ({el*1e3/S:.2f} ms/step)")
with wp.ScopedCapture() as cap: mjw.step(m, d)
wp.capture_launch(cap.graph); wp.synchronize()
t = time.perf_counter()
for _ in range(S): wp.capture_launch(cap.graph)
wp.synchronize(); el = time.perf_counter() - t
print(f"graph : {N*S/el:>12,.0f} env-steps/s ({el*1e3/S:.2f} ms/step)")
qpos = d.qpos.numpy()
print("pelvis z:", qpos[0, 2].round(4), "nan:", bool(np.isnan(qpos).any()))
assert not np.isnan(qpos).any() and qpos[0, 2] > 0.6
