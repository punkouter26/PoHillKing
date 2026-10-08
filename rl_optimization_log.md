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

## 2026-10-07 (late) — pool rework, torque limits, performance caveat

- **Pool:** 8 boxes on a hidden shelf → 4 boxes pinned in the sky (pose restored, velocity zeroed every control step in training, every physics step in Unity). Removes ~30 permanent contacts and 24 DoF per world. Floor is a plane again (about 25% cheaper than a box slab in Warp); boxes must never be parked under a plane.
- **Torque limits:** the Unity plugin has no `actuatorfrcrange` on joints. The same limits (88/139/50/25/5 Nm) are now actuator `forcerange`. Passive hold unchanged.
- **`ls_iterations` = 10** in the MJCF; Unity mirrors it by writing `model->opt.ls_iterations` after scene init (the plugin cannot import it).
- **Env smoke test:** `python -m koth.env` runs, 4096 envs, zero NaN, zero falls under zero action for 2 s. `warp.stream_from_torch` raised "unknown stream"; replaced by explicit `torch.cuda.synchronize()` / `wp.synchronize_device()` around the captured graph.
- **Performance is currently unmeasurable.** The stock menagerie `scene_mjx.xml` benchmark fell from ~415k to ~49k sim-steps/s at the same world count between 20:30 and 23:20, so the 10x slowdown is the machine, not the model. Observed at the time: Docker's WSL VM holding 14 GB (3 GB RAM free), a PoRace.exe build and a Unity editor on the GPU, GPU reporting a software power cap. Re-benchmark on a quiet machine before choosing `num_envs`.
- **Not yet verified in Unity:** import of `<general>` actuators with gainprm/biasprm/forcerange, armature, frictionloss, solreflimit, condim. `ModelDumpCheck` is the gate for all of these.

## 2026-10-08 — Phase B gates passed; three Unity plugin traps

- **GPU power:** the laptop was enforcing a 15 W cap on the GPU (180 MHz under load). After the power profile change the limit reads 140 W and the env runs at ~21–27k control steps/s with 4096 envs.
- **Unity licence:** batch mode needs a headless entitlement the account does not have; all automated runs use a normal editor launch with `-executeMethod ... -kothExit`. The editor must be signed in.
- **Trap 1, ASCII STL false positive:** the plugin treats any STL whose 80-byte header starts with "solid" as ASCII and aborts. 27 of 51 menagerie meshes are binary with such a header; headers rewritten in place.
- **Trap 2, default-class array tails:** the importer re-saves the MJCF through MuJoCo, whose writer truncates `biasprm` in a child default when the tail equals the parent's. Knee kv was read as 0 and the robot fell in under a second. Fixed by making the base class (0, 0) so every group writes its full pair; compiled training model is bit-identical to before.
- **Trap 3, `MjActuator.OnSyncState` zeroes ctrl:** after every `mj_step` each actuator component copies its own `Control` field into `mjData.ctrl`. Writing ctrl only on policy ticks left 9 of 10 physics steps with zero targets. `PolicyRunner` now rewrites the held targets in `preUpdateEvent` before every step.
- **Result:** `ModelDumpCheck` 0 mismatches (masses, inertial offsets, joint ranges, damping, armature, frictionloss, kp/kv, ctrl and force ranges, contact masks, friction, sizes, solver options). Hold trace Unity vs Python agrees within 0.1 mm at all five sample times, both scenes.

## 2026-10-08 — R0 training runs

| run | change | outcome |
|---|---|---|
| r0_a | first reward set, termination −100·dt, no clip | episode length fell 25 → 4.8 steps while reward rose: dying early beat the per-step penalties |
| r0_b | sum clipped at 0, alive +1, termination −1 | never falls (episode 967/1000) but the clipped sum was 0 on every step, so only the entropy bonus trained; action std grew 0.5 → 2.5 (thrashing) |
| r0_c | positive terms dominate (tracking 1.75, alive 0.5, upright 0.5), smaller penalties, std capped at 1.0, entropy 0.002 | in progress: reward 13, episode 600+, std 0.36 at iteration 300 |
