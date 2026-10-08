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
| r0_c | positive terms dominate (tracking 1.75, alive 0.5, upright 0.5), smaller penalties, std capped at 1.0, entropy 0.002 | stands, std 0.36, but only 47 % of 2 m/s impacts survived at iteration 300: a stand-only policy has no stepping skill. Stopped at 400 and used as the warm start for R1 |

## 2026-10-08 — Policy parity gate, R1 runs, arena zero-shot

- **Policy gate (r0_c/300, flat):** replay max action error 1.8e-7 (bar 1e-4). Closed loop 5 s: Unity pelvis within 0.4 micron of the CPU MuJoCo reference, max obs error 4e-5, inference 0.22 ms on CPU. The reference is recorded in CPU MuJoCo driven by the same ONNX through onnxruntime (`scripts/record_reference.py`), not in Warp, so Unity is compared against the identical engine.
- **Decision: R0 is judged on the walking policy with a zero command.** `r1_a` (warm-started from r0_c/400, commands on, 10 % zero) at iteration 1000: R0 bar passes with 97.5-98 % of impacts survived (2 m/s kicks every 5 s, 6 m/s boxes), 86-88 % of robots never fall in 20 s.
- **R1 tracking metric:** error is measured on the 1 s moving average of body velocity and only when the last impact, command change and reset are more than 1.5 s old. The raw per-step error is 0.25 m/s even when standing still, purely from the kicks.
- **r1_a, iteration 1000-1200:** linear tracking 0.82 m/s for a 1.0 command, error 0.165 m/s (bar 0.15). Yaw commands ignored (commanded 0.8 rad/s, achieved 0.0), error 0.32 rad/s (bar 0.2). Cause: the gait makes instantaneous yaw rate oscillate with std 0.3-0.4 rad/s, which swamps the exp-kernel yaw-rate term.
- **r1_b:** adds `tracking_heading` (weight 1.0), the error between heading and the leashed integral of the commanded yaw rate. Resumed from r1_a/1200.
- **Design change for the arena rungs:** the policy input stays at 103. For R2 and R3 the three command slots are filled by a fixed geometric law (`obs.goal_command`, mirrored by `GoalCommand.cs`): walk toward the plateau centre (R2) or toward the opponent (R3), speed = clip(distance - stop, 0, 0.8), yaw rate = clip(2 x bearing, -1, 1). This replaces the 109/130-dim observations in the Step 2 blueprint. No network surgery and no change to Unity's ObsBuilder.
- **Arena zero-shot with r1_a/1200:** R3 passes (512 pairs x 2 seeds: 100 % meet within 6 s, 0 self-ejections). R2: plateau hold 100 %, return from the slope (spawn r > 2.3 m) 30-33 %, so R2 needs arena fine-tuning.
- **R3 Unity gate (two robots, two policy runners):** replay 5e-7. Closed loop over 6 s: exact for the first 10 ticks, then chaotic drift through foot contacts; final pelvis error 1.6 cm after 1.2 m of walking, ctrl difference 3-6 %, both upright, same end separation. Passes the 10 cm / 10 % bars.
- **Env throughput at full GPU power (4096 worlds):** flat 1 robot about 37k policy steps/s; arena 1 robot about 100k world sim-steps/s; arena 2 robots about 99k world sim-steps/s (20k policy steps/s) while sharing the GPU with a training run.

## 2026-10-08 (midday) — R1 and R2 pass, attacker stage 1, Unity duel demo, scope change

- **R1 final (`r1_b/model_final`, 2700 iterations):** R0 bar 99.4 % impact survival; R1 bar tracking error 0.14 m/s and 0.10 rad/s, 97.8 % survival, 3/3 seeds. Yaw response 0.69 rad/s for a 0.8 command (was 0.0 before the heading term). Unity gates: standing drift 1.2 mm in 5 s, walking 3.7 cm after 2.1 m, ctrl difference 2.6 %.
- **R2 (`r2_a/model_2900`, 200 arena iterations from R1):** plateau hold 100 %, return from slope spawns 93-100 %, 3/3 seeds. Unity gate from a slope spawn at r = 2.4 m: 3.2 cm final error after a 2 m climb, ctrl difference 7.9 %.
- **Scope change from the user (2026-10-08):** all agents start on top of the hill and nobody is asked to climb. `default_cfg("r2")` now spawns on the plateau only; slope spawns remain as `eval.py --slope`. The slope result above is kept for the record.
- **R4 experiments:**
  | run | setup | outcome |
  |---|---|---|
  | probe | untrained ramming (walker + attack command) vs centre-holding walker | 0 ejections in 1024 duels |
  | r4_a | symmetric self-play, alive 0.5 / upright 0.5 / win 10 | stalemate in 130 iterations: a 20 s draw paid 40, a win 10 |
  | r4_b | win 20, lose -20, draw -10, smaller survival terms | 8-30 % of rounds decided, defence learns faster than attack |
  | r4att_a | learner vs frozen walker that walks back to the centre | learner stops falling, zero wins |
  | r4att_b | same, reward on opponent radius instead of ring advantage (which is flat while pushing from inside) | zero wins in 110 iterations |
  | r4att_c | defender = frozen walker with a zero command near the rim | **98.7 % ejections, median 2.0 s, attacker loses 1.3 %** (768 duels). Control: that defender drifts out unaided in 10 % of 15 s rounds |
  | r4sp_a | mirror self-play from r4att_c/500, draw = loss = -10, win 30 | starts at 65-87 % decided, falling to about 40 % by iteration 110 |
- **Why a centre-holding defender cannot be shoved out:** same mass, same friction limit, and a controller that already survives 2 m/s kicks. Winning needs toppling or out-manoeuvring, not pushing.
- **GPU memory:** two 4096-world trainings plus an evaluation fill 12 GB and slow everything about 100x. Run one training at a time; evaluations at 256 worlds fit beside one training.
- **Unity:** `ObsBuilder.FillCombat` mirrors `obs.build_combat` (9 values), `PolicyRunner.policyObsDim` selects 103 or 112 inputs, `DemoDirector` runs duel rounds with mjData resets and a HUD. `Demo_duel.unity` is built by `PoKingHill/Build duel demo scene`. The combat block has no parity gate yet.
