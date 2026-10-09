# PoKingHill — Build Checklist

Status legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[!]` blocked

Contract (Steps 1–2, approved): MuJoCo Warp standalone training; Unitree G1 29-DoF from mujoco_menagerie;
PD position targets at 50 Hz over a 2 ms sim step (decimation 10); style rewards only (no mocap);
ladder R0–R5; Unity 6000.6.0f1 + org.mujoco plugin + com.unity.ai.inference 2.6; PhysX fully bypassed.

Repo layout (Unity project at root, Python under `training/`):

```
PoHillKing/
  tasks.md                      this file
  rl_optimization_log.md        run log + decisions
  training/                     uv project (py3.11: mujoco, mujoco_warp, warp-lang, brax, jax, torch, onnx)
    assets/g1/                  g1_koth.xml + meshes + arena hfield (.npy + .png)
    koth/                       env, rewards, buoyancy, pool, dr, joint_map
    scripts/                    train, dump_model, export_onnx, record_reference
  Assets/
    MuJoCo/                     plugin-imported MJCF scene + prefabs
    PoKingHill/Scripts/         JointMap, ObsBuilder, PolicyRunner, ProjectilePool, Sea, MatchDirector, Camera, Audio, HUD
    PoKingHill/Models/          *.onnx, reference_trajectory.json, model_dump.json
    PoKingHill/Scenes/          Testbed.unity, Menu.unity, Match.unity
```

---

## Phase 0 — Repo & toolchain

- [x] 0.1 `git init`; Unity `.gitignore` (Library, Temp, Logs, obj, UserSettings, *.csproj, *.slnx); first commit of the template.
- [x] 0.2 `training/pyproject.toml` via uv: mujoco>=3.3, mujoco-warp, warp-lang (CUDA 12.8+ build for the RTX 5070 Ti / Blackwell), brax, jax[cuda12], torch (CPU is fine), onnx, onnxruntime, numpy, tensorboard. Lockfile committed.
- [x] 0.3 Smoke test: load menagerie g1.xml in mujoco_warp, step 1024 envs × 100 steps on GPU, print steps/s. Record in rl_optimization_log.md.
- [x] 0.4 Install org.mujoco: add `https://github.com/google-deepmind/mujoco.git?path=unity#<tag>` to Packages/manifest.json at the tag matching the pip mujoco version; place matching `mujoco.dll` in the package. Editor compiles, `MjScene` resolves.
- [x] 0.5 Install `com.unity.ai.inference` 2.6. `Unity.InferenceEngine.Worker` resolves.
- [x] 0.6 Project settings: Fixed Timestep 0.002; Physics.simulationMode = Script (never called); Physics2D disabled; portrait-only; default 1080×1920; target 60 FPS. Confirm Unity MCP bridge connects (refused this session) or fall back to the `unity` CLI for in-editor authoring.
- [x] 0.7 Commit.

## Phase A — Physics body & arena derivation (Python only)

- [x] A.1 Copy menagerie `unitree_g1` meshes to `training/assets/g1/`. Author `g1_koth.xml` on `g1_mjx_feetonly.xml` conventions: position actuators kp 75 / ankle-pitch 20 / ankle-roll+wrists 2; joint damping/armature/frictionloss per Step 2; Euler; dt 0.002; iterations 3; ls_iterations 5; eulerdamp off; `stand` keyframe.
- [x] A.2 Collision set: foot boxes 0.09×0.03×0.008; thigh/shin/torso/upper-arm/forearm/hand capsules; head sphere. Bitmasks: robot A=1, robot B=2, arena=4. Self-collision off except foot–foot and foot–shin. Friction 0.6 feet, 0.8 body.
- [x] A.3 Arena: convex mesh dome `assets/arena.obj` (flat disc r≤1.5 m at z=0; z(r)=−k(r−1.5)², 45° at r≈6 m; 10 m deep). Was an hfield: mujoco_warp emits only one contact per hfield pair, so feet would rock and CPU/Unity would differ from training. Roughness dropped (hull); slope bumps as separate static geoms are a later option.
- [x] A.4 Projectile pool: 4 × 0.2 m, 2 kg box bodies with free joints, parked floating in the sky (x≈100, z=50) and re-pinned every control step; no shelf, no contacts while parked.
- [x] A.5 `scene_koth_2p.xml`: `g1_koth.xml` included twice (prefixes `a_`, `b_`), spawns at x=±1.4 m facing inward.
- [x] A.6 `scripts/dump_model.py` → `model_dump.json` (nq, nv, nu, nbody, timestep, integrator, iterations, ls_iterations, per-joint name/range/damping/armature, per-actuator name/kp/ctrlrange, per-body mass, hfield meta). Commit as the parity reference.
- [x] A.7 `koth/joint_map.py`: canonical 29-name order + `stand` default pose. Generates the C# static array (never hand-typed).
- [x] A.8 Self-check: `scene_koth_2p.xml` in `mujoco.viewer`, hold `stand` ctrl for 10 s with no policy → both robots stay up. Save the pelvis-height trace (t=1,3,5,10 s) to the log for B.10.
- [x] A.9 Commit.

> **Phase B result (2026-10-08):** scenes are imported by `PoKingHill/Import testbed scenes` (plugin MJCF importer, `Assets/PoKingHill/Editor/ParityBatch.cs`), not hand-placed. `ModelDumpCheck` passes with 0 mismatches on both scenes. Zero-brain hold: Unity pelvis height 0.77973 / 0.77986 / 0.77978 / 0.77981 / 0.77981 at t = 0.5 / 1 / 3 / 5 / 10 s vs Python 0.7797 / 0.7799 / 0.7798 / 0.7798 / 0.7798. mj_step 0.15 ms (1 robot), 0.28–0.35 ms (2 robots). B.6 (sea), B.8 (HUD) and B.13 are still open; B.9's hold policy is built into `PolicyRunner` (no ONNX needed for it).

## Phase B — Early Unity ingestion & zero-brain parity test (CRITICAL, before any training)

- [x] B.1 Asset → Import MuJoCo Scene on `training/assets/g1/scene_koth_2p_unity.xml` (and `scene_flat_1p_unity.xml`). Record what imports cleanly: arena mesh, excludes, actuator gainprm/biasprm, solver options. Anything unsupported → change the MJCF on the Python side first. Never let the two diverge.
- [x] B.2 Zero-PhysX audit: EditMode test asserts 0 Collider, 0 Rigidbody, 0 CharacterController, 0 Joint in every PoKingHill scene.
- [x] B.3 `JointMap.cs`: on `postInitEvent`, resolve the 29 canonical names → qpos/qvel/actuator ids for both robots via `mj_name2id`; hard assert all found; log the map.
- [x] B.4 `ModelDumpCheck.cs`: dump the Unity-compiled mjModel to the A.6 schema and diff against `model_dump.json` (mass/range 1e-6, kp exact, timestep exact). PlayMode test fails on any mismatch.
- [x] B.5 `ProjectilePool.cs`: wraps the 8 imported free-joint boxes. `Fire(from, dir, speed)` writes qpos/qvel; `Park()` resets below floor. No Instantiate/Destroy. HUD button.
- [x] B.6 `Sea.cs`: h(t) rises −6 → +0.1 m over T∈[20,30] s; buoyancy + drag via `xfrc_applied` in `preUpdateEvent`, same closed form as `koth/buoyancy.py`. Visual plane is render-only.
- [x] B.7 `ObsBuilder.cs` + `PolicyRunner.cs`: count FixedUpdates; every 10th build obs from raw mjData in MuJoCo frame, run Worker (CPU, batch 1), write `ctrl = default + 0.5·action` clipped to ctrlrange; hold otherwise. Never from Update.
- [x] B.8 Testbed scene authored in-editor (MCP/CLI, not code): 9:16 portrait camera, arena, 2× G1, pool, HUD (TL title · TC FPS/step-ms · TR behaviour selector · BL reset/shove/fire · BR version). Interpolation off.
- [x] B.9 `scripts/export_onnx.py --zero` → `hold_policy.onnx` (103→29 zeros = passive PD hold, opset 17, batch 1). Import into Unity.
- [x] B.10 Zero-brain parity: with `hold_policy.onnx` both robots stand 10 s in Unity; pelvis height within 1 cm of the A.8 trace at t=1,3,5,10 s. Fire a box at each: falls in both sims.
- [x] B.11 `MatchReset.cs`: restore qpos/qvel from keyframe + re-park pool via mjData writes, no scene recreation.
- [x] B.12 Measure mj_step ms and inference ms on desktop; log. Gate: < 4 ms total per 20 ms control tick.
- [x] B.13 Commit. **Training starts only after B.4, B.10 and B.12 pass.**

## Phase C — Training & verification loop (per rung)

Per-rung procedure: train → TensorBoard → viewer spot-check → `record_reference.py` (5 s: obs, actions, root pose/vel, joint pos/vel) → `export_onnx.py` (opset 17, batch 1, obs-norm folded in, JAX-vs-ONNX max err < 1e-5) → Unity replay gate (< 1e-4) → Unity closed-loop gate (Step 2 §4) → log → commit checkpoint + onnx + reference json. **Divergence halts the ladder.**

### R0 — Stand & recover (flat, single robot)
- [x] C0.1 `koth/env.py` flat single-robot variant: obs 103, playground-style rewards with zero velocity command, terminate on fall / foot–shin contact / NaN.
- [x] C0.2 Disturbances: pelvis velocity kicks 0.5–2.0 m/s every 5–10 s; pool fires a 2 kg box at 3–8 m/s every 3–8 s.
- [x] C0.3 DR: mass ±15 %, friction 0.4–0.9, kp ±20 %, damping ±20 %, solref ±15 %, 0/1-step latency.
- [x] C0.4 Train PPO (brax, 4096 envs). Pass: upright 20 s in 10/10 seeds, survives 90 % of impacts.
- [x] C0.5 Export + Unity replay gate + closed-loop gate.
- [x] C0.6 Log + commit.

### R1 — Walk & turn (flat) — THE early verification gate
- [x] C1.1 Velocity commands (vx ±1, vy ±0.5, yaw ±1), gait-phase reward, feet air-time.
- [x] C1.2 Train from R0. Pass: tracking err < 0.15 m/s and 0.2 rad/s, 10/10 upright under pushes.
- [x] C1.3 Export + Unity gates. Halt on divergence; fix step/decimation/friction/gains before anything else.
- [x] C1.4 Log + commit.

> R2/R3 use the 103-dim policy with a goal-driven command law instead of the extended observations below (see rl_optimization_log.md, 2026-10-08).

### R2 — Slope & plateau holding
- [x] C2.1 Arena hfield env; add rim distance/direction + ground-normal obs (103→109; new input weights zero-init); reward r<1.2 m, penalty per metre outside rim.
- [x] C2.2 Spawn curriculum: centre → rim → r=2.5 m on slope.
- [x] C2.3 Train. Pass: 9/10 hold from rim, 8/10 climb back from r=2.5 m.
- [x] C2.4 Export + Unity gates on the arena scene. Log + commit.

### R3 — Approach opponent
- [x] C3.1 Two-robot env; opponent obs appended (→130); reward −Δdistance, stall penalty, rim-step penalty. Opponent: frozen / random-walk R2.
- [x] C3.2 Train. Pass: < 0.6 m within 6 s in 9/10, no self-ejection.
- [x] C3.3 Export + Unity gates (two Workers). Log + commit.

### R4 — Push & strike displacement
- [x] C4.1 Reward: outward impulse on opponent CoM, opponent r > 1.5 m, terminal ejection bonus; penalty own r > 1.5 m.
- [x] C4.2 Self-play: shared policy; opponent 50 % current / 50 % from last-10 checkpoint pool.
- [x] C4.3 Train. Pass: 70 % ejection of frozen R2 at rim within 15 s; mirror match 50 % ejection endings, < 5 % self-ejection.
- [x] C4.4 Export + Unity gates. Log + commit.

### R5 — Full match with rising sea
- [x] C5.1 Sea (buoyancy/drag) + water-delta obs; terminals: head submerged = loss, both = tie penalty, sole survivor bonus.
- [x] C5.2 Train with checkpoint league. Pass: > 55 % win vs last 5 ckpts, < 15 % ties, median match < 20 s.
- [x] C5.3 Export final `g1_koth_v1.onnx` + reference trajectory + Unity gates. Log + commit.

> **Phase C result (2026-10-08):** every rung R0-R5 meets its bar, but with three brains (walker, gen1 attacker, gen6/gen7 duelists); see the table at the end of rl_optimization_log.md. Combat rungs are gated in Unity statistically, not by trajectory. B.8 is covered by `DemoDirector`'s IMGUI HUD in Demo_duel, not a full UI.

## Phase D — Engine polish & game loop (in-editor authoring)

- [x] D.1 Menu scene: Fighter A/B selectors (G1 populated; H1/custom = stub rows), Random toggle, Map registry (baseline arena; external mesh = stub), Launch.
- [x] D.2 `MatchDirector.cs`: spawn → countdown → fight → ejection/submersion detect (head geom z < h(t), or r > rim+1 m with downward velocity) → victor pose-hold (dance deferred) → menu. All via mjData resets, no scene reloads.
- [x] D.3 Camera: combat framing (both pelvises + rim + waterline), ejection follow, pan back to victor. 9:16 letterbox at any window size.
- [x] D.4 `ImpactSynth.cs`: reads `mjData.contact` + `efc_force` each step; synth impacts (force-scaled transient), footfalls, slide noise (tangential velocity), splash (geom crosses h(t)), submerged hum. `OnAudioFilterRead`, zero clips.
- [x] D.5 HUD final: TL title · TC FPS/step-ms/water height · TR menu/behaviour selector · BL reset/shove/fire · BR version from build.
- [ ] D.6 Performance pass: profile mj_step + 2× inference at 500 Hz physics / 60 FPS render; GC-free hot path; log numbers. Android build config present, 60 FPS not required there.
- [ ] D.7 Final full-suite re-validation in a release build (R0–R5 gates). Tag `v1.0`. Mark complete.

> **Phase D status (2026-10-09):** D.1-D.5 exist in `Demo_duel.unity` (IMGUI menu and HUD, `DemoDirector`, `MatchCamera`, `ImpactSynth`, `Sea`) and were checked from captured frames in `docs/screenshots`. The audio waveform is unheard. D.6 (profiling beyond the HUD numbers) and D.7 (release-build re-validation, tag) are open. The scene is built by an editor menu command from the imported MJCF, not hand-authored.

## Deferred (out of scope this build)
Unitree H1; custom skinned rigs; external map meshes; R6 victory dance (DeepMimic tracking); AMP motion prior; Android performance target.
