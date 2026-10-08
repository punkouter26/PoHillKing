# PoKingHill — RL Optimization Log

Chronological record of runs, measurements, and decisions. Newest at the bottom.

## 2026-10-07 — Phase 0 toolchain decisions

- **Backend:** MuJoCo Warp 3.15.0 standalone. JAX CUDA wheels are Linux-only and this machine has no WSL distro, so brax/playground PPO is out. Chosen: native Windows `mujoco_warp` + PyTorch 2.9.1+cu128 + `rsl_rl` PPO. Upside: ONNX export is a direct `torch.onnx.export`, no JAX-to-torch weight conversion.
- **Hardware:** RTX 5070 Ti Laptop (12 GB, Blackwell), driver 610.78. Warp 1.18 JIT-compiles for sm_120.
- **Unity plugin:** org.mujoco 3.15.0 embedded at `Packages/org.mujoco` with `mujoco.dll` from the 3.15.0 Windows release. Inference: `com.unity.ai.inference` 2.6.1.
- **Plugin capability audit (from 3.15.0 sources):**
  - Supported: hfield (`MjHeightFieldShape`), position actuators, `MjExclude`, `MjGlobalSettings` with integrator / solver / iterations / cone / jacobian / flags.
  - Not supported: `<keyframe>`, `<contact><pair>`, `ls_iterations`, `eulerdamp` flag. Timestep comes from Unity Fixed Timestep, gravity from Unity Physics Manager.
  - **Decision:** training MJCF uses only what the plugin can mirror: geom-level friction (no pairs), eulerdamp left at default (enabled), ls_iterations left at MuJoCo default (50), Newton solver, Euler integrator, iterations 3. Default pose is written to qpos from C#, not from a keyframe.
- **Unity project settings set:** Fixed Timestep 0.002, PhysX `SimulationMode = Script` (never invoked), Physics2D auto-simulation off, portrait only, 1080×1920 default.

## 2026-10-07 — 0.3 mujoco_warp smoke benchmark (RTX 5070 Ti Laptop)

| model | worlds | eager | CUDA graph |
|---|---|---|---|
| menagerie `scene.xml` (mesh colliders, implicitfast, iters 100/ls 50) | 1024 | 13.5k env-steps/s | 226k |
| menagerie `scene_mjx.xml` (primitive colliders, dt 0.004, iters 5/ls 8) | 1024 | 38.7k | 415k |
| same | 4096 | 75k | 546k |

Decision: always step through a captured CUDA graph (`wp.ScopedCapture`), eager is launch-bound. Primitive colliders only. At 4096 worlds × decimation 10 that is ~55k policy steps/s before reward/obs overhead.
- **Arena: hfield → convex mesh.** mujoco_warp 3.15 warns `MULTICCD ... without multicontact support: HFIELD-CAPSULE, HFIELD-BOX` and generates at most one contact per hfield pair, so a foot box on the plateau would balance on a single point in training while CPU MuJoCo and Unity give four. Replaced by `arena.obj` (818 verts), a convex dome with a flat 3 m disc; MuJoCo's convex hull equals the solid, BOX-MESH pairs get full multi-contact (CPU: 4 contacts per foot, same as the flat floor). CAPSULE-MESH is still single-contact in Warp (limbs sliding on the slope); acceptable. Surface roughness is dropped for now.
- **Environment note:** Unity editors are pinning the GPU at ~99% with no Python running, which slows Warp kernel JIT and all GPU tests several-fold.
- **Two-robot arena scene in Warp:** 10 s passive hold passes for both robots (1024 worlds, z = 0.7798, zero spread), matching CPU MuJoCo. Throughput was only ~7k env-steps/s versus ~155k on the flat single-robot scene, with 730 "solver iterations limit reached (5)" warnings. Measured while Unity editors held the GPU at ~99%, so the number is a floor, not a verdict. To revisit before R2: re-measure on an idle GPU, raise `iterations`, and check mesh-collision cost.
