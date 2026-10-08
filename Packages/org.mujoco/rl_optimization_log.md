
## 2026-10-07 — Phase A: model derivation findings

- **Canonical model text:** `training/assets/g1/scene_*_train.xml` is MuJoCo's own re-saved XML (`mj_saveLastXML`, 6 significant digits) plus a keyframe; `scene_*_unity.xml` is the same text minus keyframe/sensor. Both compile to identical models (asserted in `build_mjcf.py`). Training never loads the authoring XML.
- **Collision masks instead of `<contact><pair>`** (plugin cannot import pairs): 1=robot A, 2=robot B, 4=arena, 8=box, 16=foot/shin cross-leg, 32=parking shelf. Same-leg foot/shin pairs excluded with `<exclude>`.
- **Bug found:** a parked projectile pool under an infinite floor plane gets ejected at ~950 m/s. Floor is now a finite 40×40×1 m slab; the hfield arena is finite by construction; the pool rests on a hidden shelf at z=−50 that collides only with boxes.
- **Passive stance vs PD gains** (10 s hold, gravity-compensated targets, CPU):
  | gains (hip/knee/ankle-pitch, kv) | result |
  |---|---|
  | menagerie-mjx 75/75/20, kv 2 | falls at ~2 s (stock `scene_mjx.xml` too) |
  | Unitree 100/150/40, kv 2–4 | falls at ~2 s |
  | IsaacLab-like 150/200/40, kv 5 | falls |
  | 150/200/150 | falls at ~4 s |
  | **200/300/200, kv 5** | **holds, pitch settles 1.3°** ← chosen |
  | 300/300/300, kv 8 | holds, 0.4° |
  | menagerie 500, dampratio 1 | holds, 0.1° |
  Solver iterations, contact solref, integrator, cone, and foot geometry made no difference; only chain stiffness does. Joint `actuatorfrcrange` (88/139/50/25 Nm) still caps torque. Hold targets are offset by the steady-state sag (`hold_ctrl`, ≤0.16 rad) and stored in the keyframe `ctrl`.
- **mujoco_warp buffers are per world:** `put_data(nconmax=64, njmax=400)`. The default budget (derived from the initial `mjd.ncon`) silently drops contacts once the pool lands on its shelf → robot fell through the floor. Oversizing (`njmax=160·N`) OOMs on 12 GB.
- **Parity check #0:** CPU MuJoCo and Warp agree to 4 decimals on the pelvis-height trace over 10 s (std across 1024 worlds = 0).
- **Unity plugin on 6000.6:** `org.mujoco` 3.15.0 fails to compile (`GetInstanceID()` is error-level obsolete in Unity 6.6). Patched to `GetEntityId()` in `MjScene.cs` and `MjMeshFilter.cs` inside the embedded package.
