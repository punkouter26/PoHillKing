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

## 2026-10-08 (afternoon) — combat-input Unity gate, league training

- **Unity gate for the 112-input attacker (duel, a_ = attacker, b_ = standing walker):** replay max action error 1.7e-6 with all 112 inputs. Closed loop: tick 0 and tick 1 agree to 2e-6 including the 9 combat values (so `ObsBuilder.FillCombat` matches `obs.build_combat`), then chaotic growth during a 1 m/s sprint and a collision. Up to the ejection (2.2 s): pelvis within 4.8 cm (attacker) and 7.0 cm (defender), both upright, same outcome (defender leaves the ring). Defender ctrl difference 5 %; attacker 16 %, over the 10 % bar. Verdict: inputs and network verified; trajectory equivalent in outcome but not within the strict ctrl bar once contact begins.
- **Mirror self-play `r4sp_a`** (draw = loss): decided rounds 65-87 % at the start, 40-52 % after 110 iterations. Stopped.
- **League (`scripts/league.py`, rung `r4league`):** generation k is warm-started from k-1 and trained as robot a_ against a per-world random draw from the previous five generations, frozen, as robot b_. Both sides use the attack command. gen1 = `r4att_c/500`. After each generation: 3 seeds x 256 duels x 25 s against its pool, written to `runs/league/results.jsonl`. R5 bar per generation: win share of decided rounds > 55 %, ties < 15 %, median winning time < 20 s.
- **R5 definition under the summit-only rule:** leaving r < 1.7 m or falling already ends the round, so the rising sea only acts as the deadline (20-30 s) that turns a standoff into a tie. Buoyancy and drag apply only to bodies that have already left the summit and are a Unity presentation item, not a training input.

## 2026-10-08 (evening) — league results, Unity sea, statistical parity

- **Noise handicap fixed:** frozen opponents now sample actions with their own exploration std during training. Direct check with identical policies: both noisy 51/49, both clean 49/51, noisy vs clean 35/65.
- **League results** (3 seeds x 256 duels x 25 s, deterministic, against the pool of earlier generations):
  | gen | pool | win | loss | tie | win share of decided | median win time | R5 bar |
  |---|---|---|---|---|---|---|---|
  | 2 | 1 | 33.9 % | 66.2 % | 0 % | 33.9 % | 3.1 s | fail |
  | 3 | 1-2 | 63.3 % | 33.0 % | 3.8 % | 65.8 % | 3.1 s | pass |
  | 4 | 1-3 | 63.0 % | 30.9 % | 6.1 % | 67.1 % | 2.7 s | pass |
  | 5 | 1-4 | 49.9 % | 47.5 % | 2.6 % | 51.2 % | 2.6 s | fail |
  | 6 | 1-5 | 57.2 % | 39.6 % | 3.3 % | 59.1 % | 3.3 s | pass |
  Not monotonic (gen5 dipped), but gen6 clears the bar against the full five-generation pool.
- **gen6 against itself:** 94-98 % of rounds decided, wins split about evenly, median 5.2 s. R4 mirror bar (>= 50 % decided) met.
- **Forgetting:** gen6 against the standing walker at the rim wins only 4-7 % and loses 65 % (gen1 won 98.7 %). League play against chargers alone drops the skill of dealing with a stationary opponent. gen7 trains against a pool that includes the standing walker twice plus gens 1, 3, 4, 5, 6 (`path.pt:stand` entries in `--opponent`).
- **Unity:** `Sea.cs` (rising water as the round deadline, buoyancy and drag through `xfrc_applied` on robot bodies below the surface, render-only disc with its collider removed), `DemoDirector` now attacker vs attacker with the sea, 60 FPS cap. `Demo_duel.unity` rebuilt by the menu item.
- **Trajectory parity is the wrong gate for adversarial rounds.** Attacker vs attacker in Unity: replay 1e-6, tick 1 at 2e-6, then 22 cm and 57 cm apart after 3 s because each robot reacts to the other. Replaced by a statistical gate: N rounds in Unity (`ParityBatch.RunDuelStats`) against N rounds in CPU MuJoCo with the same rules (`scripts/duel_stats.py`), comparing decided share and winning-time quartiles. Python, gen3 vs gen3, 200 rounds: 53.5 % decided, median 3.4 s, quartiles 2.9 / 4.0 s.

## 2026-10-08 (night) — statistical Unity parity, generation 7, ladder summary

- **Statistical parity, gen3 vs gen3, rules of `DemoDirector` (no sea, 25 s bell):** Unity 60 rounds: 53 % decided, winning time median 3.32 s, quartiles 2.89 / 3.64, mean 3.87. CPU MuJoCo 200 rounds: 53.5 % decided, median 3.38 s, quartiles 2.94 / 4.02, mean 3.92. The unfocused editor renders under 1 frame per second; stats mode sets `maximumDeltaTime = 2` and `timeScale = 50` so each frame carries up to 1000 physics steps.
- **gen7** (450 iterations; pool = standing walker x2, gens 1, 3, 4, 5, 6): against gens 2-6 wins 72.5 %, loses 17.8 %, ties 9.6 % (80 % of decided, median 4.3 s), R5 bar passes 3/3. Against itself only about 50 % of rounds are decided. Against the standing walker at the rim: 6.6 % wins, 31 % losses, 62 % standoffs, so the anchor did not restore that skill.
- **Which brain meets which bar:**
  | bar | brain | result |
  |---|---|---|
  | R0, R1 | walker `r1_b` | 99.4 % impacts survived; tracking error 0.14 m/s, 0.10 rad/s |
  | R2 | `r2_a/2900` | plateau hold 100 %, slope return 93-100 % (slope part shelved by the user) |
  | R3 | walker + goal law | 100 % of pairs meet within 6 s, no self-ejection |
  | R4 eject a standing opponent at the rim | gen1 (`r4att_c/500`) | 98.7 %, median 2.0 s |
  | R4 mirror, rounds decided | gen6 | 94-98 % |
  | R5 win share vs last five, ties, time | gen7 | 80 % of decided, 9.6 % ties, 4.3 s |
- **Not met by one brain:** no single policy passes both the standing-opponent bar and the league bar. The separate "self-ejection under 5 %" clause of the R4 bar was not measured on its own.
- **Unity:** `attacker_policy.onnx` is now gen7. The statistical gate above was run with gen3.

## 2026-10-08 (late) — one combat brain for all bars, Unity camera and audio

- **gen7 statistical parity:** Unity 60 rounds 15 % decided, mean winning time 13.2 s; CPU MuJoCo 200 rounds 14.5 %, 13.0 s. Parity holds, and it shows gen7 is passive against itself.
- **gen8** (from gen7, half standing opponents): draws kept rising; stopped at iteration 220.
- **allround1** (700 iterations from gen1, pool = standing walker x6, gens 1, 3, 4, 5, 6, 7). Starting from the brain that already beats a standing opponent kept that skill alive while it learned to duel:
  | bar | result |
  |---|---|
  | standing walker at the rim (>= 70 % ejections) | 93.4 % (89.5 / 94.9 / 95.7), median 2.0 s |
  | self-ejection against that opponent (< 5 %) | 6.6 % (10.5 / 5.1 / 4.3): narrow miss |
  | vs gens 3-7 (> 55 % of decided, < 15 % ties, < 20 s) | 87 % of decided, 3.6 % ties, 2.5 s |
  | vs itself (>= 50 % decided) | 86 % decided, median 5.1 s |
- **allround2** (400 more iterations, pool adds allround1): beats allround1 head to head 68 / 27, but is worse elsewhere: standing walker 87.4 % with 12.6 % self-ejection, gens 3-7 78 / 20. Not promoted.
- **Unity statistical parity, allround1 vs itself:** Unity 100 rounds 96 % decided, winning time median 8.4 s (quartiles 6.2 / 13.7); CPU MuJoCo 200 rounds 96 % decided, median 7.3 s (5.4 / 12.1). `attacker_policy.onnx` is allround1.
- **Unity presentation:** `MatchCamera` (combat framing, follows the loser off the summit, returns), `ImpactSynth` (thuds from pelvis velocity jumps, footfall clicks, splash and bubbles at the waterline, all synthesized in `OnAudioFilterRead`, no audio files). Both compile and run without exceptions in an 8-round editor run; neither has been looked at or listened to.
- **Three-hour champion league started 19:08** (`scripts/allround_league.py`): each generation warm-starts from the champion (allround1), trains 350 iterations with `self_edge = -5` against standing walkers, gens 1 / 6 / 7 and recent all-rounders, and is promoted only if it beats the champion, keeps >= 85 % against the standing walker, and does not self-eject more.

## 2026-10-08 19:08-21:57 — three-hour champion league (user request)

`scripts/allround_league.py --hours 3`, 350 iterations per generation, `self_edge = -5`, each generation warm-started from the champion.

| generation | from | standing walker: win / self-out | vs gens 3-7: share of decided / ties | vs champion: share / ties | promoted |
|---|---|---|---|---|---|
| allround3 | allround1 | 92.6 % / 7.4 % | 76.4 % / 2.5 % | 80.5 % / 15.8 % | yes |
| allround4 | allround3 | 92.7 % / 7.3 % | 78.2 % / 0.9 % | 59.0 % / 11.5 % | yes |
| allround5 | allround4 | 83.6 % / 16.4 % | 68.0 % / 0.8 % | 33.8 % / 5.6 % | no |
| allround6 | allround4 | 82.0 % / 18.0 % | 62.8 % / 0.5 % | 27.8 % / 11.6 % | no |
| allround7 | allround4 | 88.7 % / 11.3 % | 77.3 % / 0.8 % | 57.8 % / 20.1 % | no (self-out) |

- **Champion: allround4.** Against itself 89 % of rounds decided (median 5.5 s). Against allround1: wins 74.5 %, loses 15.9 % (82 % of decided).
- **Self-ejection did not improve.** 7.3 % against the standing walker (allround1: 6.6 %, bar 5 %). The stronger rim penalty made no measurable difference; three later generations were worse (11-18 %).
- **Strength against the older Duelists (gens 3-7) is lower than allround1's** (78 % vs 87 % of decided) while head-to-head strength rose. The league is non-transitive: beating the current champion is not the same as beating everything before it.
- **Unity statistical parity, allround4 vs itself:** Unity 100 rounds: 96 % decided, winning time median 6.75 s, quartiles 4.84 / 9.70, mean 7.34. CPU MuJoCo 200 rounds: 99.5 % decided, median 6.84 s, quartiles 4.81 / 9.15, mean 7.62. `attacker_policy.onnx` is allround4.

## 2026-10-08 (after the league) — what the "self-ejection" figure really was

The 7.3 % reported for the champion against the standing walker was every round the attacker lost, not self-ejection. Broken down over 1024 duels (allround4, first round each):

| how the attacker went out first | rounds | share |
|---|---|---|
| left the ring (radius > 1.7 m) | 13 | 1.3 % |
| pelvis below 0.3 m: tripped or a failed lunge, typically 1.4 s in and 0.8 m from the opponent | 39 | 3.8 % |
| tipped over | 2 | 0.2 % |
| own foot touched own shin (a training termination rule, not a game rule in Unity) | 17 | 1.7 % |

- **R4 clause as written** ("against itself, at least 50 % of rounds end by ejection, under 5 % self-ejection"), measured directly on allround4 vs itself, 1024 duels, 25 s: 65.9 % of rounds end by a ring-out; self-ejection (ring-out with the opponent more than 0.9 m away) 0.2 %; 14.1 % end by a fall; 8.3 % by the own-leg-contact rule; 11.7 % undecided.
- So the clause passes: 65.9 % ejection endings, 0.2 % self-ejection in the mirror, 1.3 % against the standing walker. The earlier "narrow miss" was a conservative proxy of mine.
- **Rules mismatch to keep in mind:** training and `eval.py` end a round on own foot-shin contact; Unity's `DemoDirector` and `duel_stats.py` do not. That is why the mirror "decided" share differs between `eval.py` (88-89 %) and the Unity-rule statistics (96-99.5 %).
- **Rim brake tried and not adopted:** an optional term in `obs.goal_command` that slows the commanded speed near the rim (`--rim-brake rim gain vmin`). Two settings on allround4 left the loss rate at 7 %, consistent with the breakdown above (most losses are falls, not overshoot). It stays off by default and is not mirrored in Unity.

## 2026-10-08/09 — careful1 becomes champion; Unity menu and visual verification

- **careful1** (400 iterations from allround4; pool = standing walker x6, allround4, allround1, gen7, gen6; `termination = -25`, `self_edge = -5`). Making a loss cost 2.5 times a draw, with standing opponents in 60 % of rounds, cut the falls without making it passive:
  | bar | result |
  |---|---|
  | standing walker at the rim, 5 seeds x 256 | 97.0 % ejected (95.7-98.4), median 1.9 s |
  | all losses against that opponent (strict reading of "self-ejection < 5 %") | 3.0 % (1.6-4.3), every seed under 5 % |
  | vs gens 3-7 | wins 90.1 %, loses 9.0 %, ties 0.9 % (91 % of decided), median 2.4 s |
  | vs allround4 (previous champion) | 39.7 % / 43.1 % / 17.2 % ties: an even match |
  | vs itself | 92 % of rounds decided, median 5.6 s |
  `runs/league/champion.txt` = careful1; `attacker_policy.onnx` = careful1.
- **Unity statistical parity, careful1 vs itself:** Unity 100 rounds: 99 % decided, winning time median 7.28 s (quartiles 5.58 / 11.48, mean 8.43). CPU MuJoCo 200 rounds: 98.5 % decided, median 8.40 s (5.64 / 12.16, mean 9.16).
- **Unity presentation, now looked at** (`SCREENSHOTS/archive/menu.png`, `combat.png`, `ejection.png`, captured by `ParityBatch.PlayDuelDemo -kothShots <dir> -kothExit`):
  - Pre-match menu: fighter A, fighter B (All-rounder champion, Duelist gen7, Rammer gen1, standing Walker), random matchup, map, Launch. `PolicyRunner.SetBrain` swaps ONNX brains at runtime; one round per launch, then back to the menu.
  - First screenshots showed the robots standing on open water: the plugin's renderer for the arena mesh geom draws nothing (0 vertices in its MeshFilter) even though the imported mesh asset is right. An OBJ arena additionally came in lying on its side, so the arena is now written as binary STL. Fix: a render-only `ArenaVisual` object showing the same mesh asset with a two-sided material.
  - HUD labels drawn with a shadow (white text was unreadable on the sky).
  - Audio triggers never fired with single-step thresholds. Now windowed: pelvis velocity change over 20 ms for impacts, descent-then-stop for footfalls. One 20 s run: 59 impacts, 481 footfalls. The editor does not run the audio thread while unfocused, so the output waveform itself is still unheard.

## 2026-10-09 — D.6 performance pass (Windows player), report screenshots

Measured by `PerfProbe` (`-kothPerf 30`: 2 s warm-up, then 30 s of continuous champion-vs-champion rounds, writes `perf_unity.json` next to the player). Player built by `ParityBatch.BuildPlayer` into `Builds/Win` (`-kothDev` for a development build). 540x960 window, Core Ultra 9 275HX, vSync off, target 60 FPS.

| | release player (before the two fixes below) | development player (final code) |
|---|---|---|
| frame rate | 59.98 FPS, frame p99 16.8 ms, max 17.7 ms | 59.98 FPS, p99 16.9 ms, max 21.3 ms |
| physics rate | 500.0 steps/s, at most 9 steps in one frame | 500.0 steps/s, at most 11 |
| whole physics tick (2 brains + ctrl + sea + mj_step + state sync) | mean 0.23 ms, p50 0.16, p99 0.88, max 2.4 | mean 0.26 ms, p50 0.17, p99 1.06, max 2.7 |
| mj_step + plugin state sync | mean 0.19 ms, p99 0.47 | mean 0.21 ms, p99 0.56 |
| one brain inference | mean 0.22 ms, p99 0.40, max 1.6 | mean 0.26 ms, p99 0.61, max 1.9 |

- **Budget:** one 20 ms control tick costs about 10 x 0.19 + 2 x 0.22 = 2.3 ms in the release player (gate from B.12: under 4 ms). Physics uses about 12 % of real time.
- **Managed allocation inside the physics tick: 0 bytes, 0 of 15,001 steps allocate** (development player, profiler counter `GC Allocated In Frame` read before and after each tick). Two fixes got it there:
  - `PolicyRunner` read the action with `ReadbackAndClone` + `DownloadToArray` (a tensor and a float array per brain per tick). Now `CompleteAllPendingOperations` + `AsReadOnlySpan` on the worker's own output.
  - The plugin's `MjScene.FixedUpdate` created two `MjStepArgs` per step (32 bytes each); now one cached object. This is an edit to the embedded `Packages/org.mujoco`.
- **Not allocation-free: the frame outside the physics tick, about 21 KB per frame** (roughly one gen-0 collection per second, no visible hitch: worst frame 21 ms). `DemoDirector`'s HUD styles are now cached, which moved it only from 23 to 21 KB, so the rest is IMGUI itself and/or the plugin's per-component `Update`. Not pursued: the frame rate holds. A UI Toolkit HUD is the fix if it ever matters.
- **Measurement limits:** neither allocation counter works in a release player (`GC.GetAllocatedBytesForCurrentThread` reads 0 under Mono, the profiler counter is absent), so the zero-allocation figure is from the development build only. The release build was not re-measured after the two fixes. The editor was not profiled: unfocused it renders about one frame per second.
- **R1 Unity gate re-run after the readback change:** replay max action error 1.2e-6 (bar 1e-4), closed loop passes (pelvis within 3.7 cm). The other rungs' gates were not re-run; that is D.7.
- **Android:** build settings exist in ProjectSettings; no Android build was made or measured.
- **Report:** `training_report.html` now embeds three annotated TensorBoard screenshots (`SCREENSHOTS/archive/tb_*.png`, taken with headless Chrome from the legacy Scalars tab; the Time Series tab renders blank headless). `Builds/Win` currently holds the development build.

## 2026-10-09 — D.7 final re-validation, v1.0

**Training side (CPU/Warp eval, 3 seeds x 256):**

| rung | checkpoint | result |
|---|---|---|
| R0 stand | `r1_b/model_final` | pass 3/3, 98.9-99.5 % of impacts survived |
| R1 walk | `r1_b/model_final` | pass 3/3, tracking error 0.14 m/s and 0.09-0.10 rad/s, 97.4-98.0 % survival |
| R2 plateau hold | `r2_a/model_2900` | pass 3/3, 99.6-100 % hold |
| R3 approach | `r1_b/model_final` | pass 3/3, 100 % meet within 6 s, 0 self-ejections |
| R4/R5 combat (`judge.sh careful1`) | `careful1` | standing walker 98.0 % wins / 1.9 % losses, median 1.9 s; gens 3-7 89.2 % / 9.9 % / 0.9 % ties, median 2.3 s |

- R2's slope-return half was not exercised: the eval spawned no robot on the slope (`n_slope: 0`), consistent with all agents starting on the summit.
- R3 on `r2_a/model_2900` fails 2/3 (seed 0: 2.7 % of pairs with a self-ejection, bar 2 %). R3 is judged on the walker, which passes; which checkpoint `r3_policy.onnx` was exported from was not checked.
- The mirror-match "share of rounds decided" bar was not re-measured here (`judge.sh` reports win / loss / tie).

**Unity editor gates (6000.6.0f1, `ParityBatch.RunPolicy`), all exit 0:**

| rung | replay max action error (bar 1e-4) | closed loop: pelvis error / ctrl difference (bars 10 cm / 10 %) |
|---|---|---|
| R0 | 1.1e-6 | 1.2 mm / 0.1 % |
| R1 | 1.2e-6 | 3.7 cm / 2.6 % |
| R2 | 1.3e-6 | 3.4 cm / 6.9 % |
| R3 (a, b) | 5e-7 | 1.2 cm / 3.3 %, 1.4 cm / 6.3 % |

**Release player (`Builds/Win`, built by `ParityBatch.BuildPlayer`, 0 errors, 354 MB):**
- Combat statistical gate inside the player (new `-kothRounds N` switch on `DemoDirector`, same rules as `RunDuelStats`; result written next to the executable): careful1 vs itself, 100 rounds, 100 % decided (54 / 46), winning time median 7.67 s (quartiles 5.42 / 10.92, mean 8.64). CPU MuJoCo reference, 200 rounds: 98.5 % decided, median 8.40 s (5.64 / 12.16, mean 9.16). Parity holds.
- Performance (`-kothPerf 30`, 540x960), now measured on the final code: 59.99 FPS, frame p99 16.7 ms, max 17.0 ms; 500.3 physics steps/s; physics tick mean 0.23 ms, p99 0.86, max 2.3; mj_step mean 0.18 ms; one brain inference mean 0.21 ms, p99 0.35.
- The trajectory gates above ran in the editor, not the player: the parity probes live in the testbed scenes, which are not in the build.

**Still open at v1.0:** the audio output has not been listened to; no Android build was made; the allocation counters do not work in a release player, so the zero-allocation figure remains the development-build one.

**Tooling note:** there is no Unity Hub on this machine. The editor gets its licence from the Unity CLI sign-in (`unity auth login`, then `unity license activate`); without it every `Unity.exe` launch exits with code 198 and a licence dialog.

## 2026-10-09 (afternoon) — Mountain top map, second fighter body (Kim)

**Mountain top map (render only, physics unchanged).** `tools/make_mountain_map.py` runs inside Blender and writes `Assets/PoKingHill/Maps/Mountain`: a rock hill, 70 loose boulders and nine distant peaks, textured from two CC0 Poly Haven scans (`aerial_rocks_02`, `rock_face_03`; sources in the git-ignored `tools/polyhaven`). The visible summit equals the collision dome inside r = 1.9 m (asserted in the script); further out the rock differs from it by up to 0.4 m (r 3-5 m) and 1.2 m (r 5-8 m). `DemoDirector.mapVisuals` switches between "Mountain top" (default) and the plain dome. Release player rebuilt with it: 59.99 FPS, 500 steps/s, tick mean 0.22 ms. Boulders were seen in Blender only; `ParityBatch.MapOverview` (edit-mode render of the whole map) exists but its corrected version has not been run.

**Kim, a second fighter body.** From the scan `Kim.glb` (21-bone rig, not in git):
- `tools/make_kim.py` (Blender): scales her to 1.60 m, faces her along +x, lowers the arms from the T-pose and makes that the rest pose, measures limb thickness, exports `Fighters/Kim/kim.fbx` + 2k textures and `training/assets/kim/kim_rig.json`.
- `koth/build_kim.py`: `kim_mjx.xml` with the G1's 29 joint names and order (observations, network shape, env and Unity runner are shared). Joint centres are her rig's bone heads, left/right averaged; 56 kg split by de Leva (1996) female segment fractions; actuator force limits are adult-woman peak joint torques (hip 150, knee 150, ankle 110, shoulder 55, elbow 45 Nm); human ranges of motion. Her rig puts the hips low: thigh 0.27 m, shank 0.34 m.
- `KOTH_ROBOT=kim` selects `training/assets/kim` in `build_mjcf.py`, `env.py` and the scripts; the G1 stays the default and its assets were not regenerated.
- Passive hold: 10 s in mujoco_warp on the flat and the two-fighter arena scene (pelvis 0.744 m). Unity (`-kothRobot kim`): `ModelDumpCheck` passes with 0 mismatches, both fighters hold at 0.744 m. One importer trap: it ignores `mass="0"` on a geom and falls back to density (her feet came out at 1.01 kg instead of 0.57 kg), so the rounded foot edges now carry 20 g each.
- `SkinFollower.cs`: each bone of the scan follows one MuJoCo body; rest pose matches the physics zero pose within 0.1 mm. `SCREENSHOTS/archive/kim_standing.png`.
- A random import-folder suffix collided with an old folder once (`Assets/Local/MjImports/scene_koth_1p_unity404`); re-running the import fixed it.
- **Kim R0 (stand under kicks and boxes), first attempt: fails.** `kim_r0`, 300 iterations from scratch with the G1's reward and disturbance settings: training reward 6.3, mean episode 268 steps, but `eval.py --rung r0` 0/3 seeds (about 1,550 falls against 1,200 impacts per seed, no robot stays up for 20 s). The walk rung that was chained after it was stopped. Open: train longer (the G1 took 400 iterations of R0 and its bar was finally met by the walking policy), and check whether the 2 m/s kick and 6 m/s box settings and the reward's G1-tuned terms (swing height 0.12 m, pose penalties) suit a 56 kg body with short thighs. No Kim policy has been exported to Unity.

## 2026-10-09 (evening) — Agent ticks in the menu, four-fighter free-for-all

- **Menu:** one "Agents in the game" list with a tick per agent replaces the Fighter A / Fighter B pickers. One ticked = mirror match, two = those two, three or four = all of them at once. Kim is listed but locked (no brain, see above). The menu is IMGUI and reacts to mouse clicks only: in the editor's Simulator view (touch) nothing responds, so `PlayDuelDemo` now opens the Game view.
- **Four-fighter scene:** `python -m koth.build_mjcf 4p` writes `scene_koth_4p*.xml` (robots a_ b_ c_ d_, collision bits 1, 2, 32, 64) without touching any training asset; `ParityBatch.ImportOne -kothScene koth_4p` imports it. `ModelDumpCheck` passes (nbody 125, ngeom 253).
- **DemoDirector** now seats agents on `fighters[]` (first N bodies; the rest are pinned 50 m up before every step). A fighter is out when it leaves the plateau or falls; with more than two playing it then goes limp (runner paused). Last one in wins; nobody left or the sea = tie. Each brain still has one-opponent inputs: it is given the nearest fighter still in. **No brain was trained for this**; one captured four-way match was won by gen7 in 3.3 s after a first round that ended in a tie. Win rates in free-for-all were not measured.
- **Scenes:** the game (`BuildDuelDemo()`, player build, `PlayDuelDemo`) uses koth_4p; the two-fighter gates (`RunDuelStats`, `RunDuelParity`) build Demo_duel from koth_2p, the scene the brains were trained in. Both write the same `Demo_duel.unity`, so whichever ran last is on disk.
- **Two-fighter gate after the refactor:** careful1 vs itself, 100 rounds in the editor: 98 % decided (49 / 49), winning time median 8.36 s (5.87 / 11.95). CPU MuJoCo reference: 98.5 %, 8.40 s (5.64 / 12.16). Unchanged.
- **Not done:** the release player was not rebuilt; `-kothPerf` and the player's `-kothRounds` now run in the four-body scene and were not re-measured.
- `SCREENSHOTS/archive/mountain_overview.png`: the whole map from outside, boulders included (edit-mode render, `ParityBatch.MapOverview`).
- **Match camera (same evening):** stays on the fighter nearest the centre of the summit (0.3 m hysteresis), pulled back to keep the other fighters still in the round in view. A knocked-out fighter is shown only when its pelvis is within 0.5 m of the water: instant cut, held until it is 0.6 m under or 3 s have passed, newest faller takes over; then an instant cut back. The old "follow the loser down the slope" shot is gone. One capture run (`-kothWaitSplash`): cut to c_, then d_, at sea level -0.8 m, back on the time limit; in that run the fallen fighters lay near the summit and the rising sea reached them, they did not tumble into it. `SCREENSHOTS/archive/camera_cut_to_water.png`.
- **Kim in the game (same evening), on request before she is ready.** Game scene is now `koth_5p` (`python -m koth.build_mjcf 5p`): four G1 (a_..d_) and Kim (e_, collision bit 128) in one model, each with its own default classes and gains; all five hold the standing pose for 5 s in CPU MuJoCo, `ModelDumpCheck` passes in Unity (nbody 155, ngeom 277). Agents and bodies carry a kind (`g1`, `kim`) and an agent only takes a body of its kind, so there can be four G1 agents and one Kim at once.
- **Her brain:** `kim_r1` = walk rung continued from the failed stand run. `Models/kim/walker_policy.onnx` is iteration 800 of it, driven by the goal law (walk at the nearest opponent, vmax 0.8). It is below every bar: R3 test 38-43 % of pairs meet within 6 s (bar 90 %), 0-0.4 % self-ejection; training episodes last about 10.6 s of 20 s. Iteration 500 met in 1-2 %. No Unity parity gate was run for her brain. One captured five-way match: she stands and steps on the summit at 1.2 s; gen7 won that round at 12.3 s. `SCREENSHOTS/archive/kim_in_game.png`.
- **Winner panel and the sea rule (same evening).** A round that reaches the sea's arrival with several fighters still standing is no longer a tie: the one nearest the centre of the summit wins. Every round ends with a centred panel (winner, how it won, a button with a countdown), 5 s, then the main menu. The statistics gate has no sea and keeps full-time rounds as ties, as in training. The camera makes no cut-aways while the panel is up. Seen in capture runs: gen7 won on centre distance with 3 standing; Kim won one round the same way with 2 standing (`SCREENSHOTS/archive/winner_panel.png`).
- **`kim_r1` finished (1,000 iterations):** R3 test of the final checkpoint 40-43 % of pairs meet within 6 s, the same as iterations 800 and 1,000, with no self-ejection; training episode length about 10.5 s. More iterations of this recipe did not help after 800, so the next step for her is the recipe (reward terms, disturbance strength, leg-contact termination), not length. The exported brain stays iteration 800.

## 2026-10-17 — Game layer rebuilt; parity re-checked; training-side cleanup (no training run)

- **Unity game layer** (commit ec29146): IMGUI replaced by a UI Toolkit HUD, an explicit match state machine, Cinemachine broadcast shots, instant replays, hazards, puppet mode, tournament, Elo ladder and per-round telemetry. PolicyRunner, ObsBuilder and the out rule are numerically unchanged (helpers only).
- **Gates after the rewrite** (editor, this machine's RTX 2060 laptop):
  - R0 trajectory gate (`ParityBatch.RunPolicy`, flat_1p): replay max action error 1.1e-6, closed-loop pelvis error 1.2 mm, ctrl difference 0.1 %. Same as the D.7 numbers.
  - Combat statistics gate (`ParityBatch.RunDuelStats`, careful1 vs itself, 200 rounds, no sea): 99 % decided (107 / 91 / 2 ties), winning time median 8.05 s, quartiles 5.07 / 11.08, mean 8.63. CPU MuJoCo reference 98.5 %, 8.40 s (5.64 / 12.16, mean 9.16). Parity holds.
- **The game's sea floated only a_ and b_** (its body list was serialized as two prefixes); it now floats every fighter body.
- **Training side, staged only** (`training/STAGED_TRAINING.md`): FFA rung r5ffa (second-opponent block, 121 inputs), Kim recipe variants, leg-contact rule switch, stalemate term, decimation override, fatigue model, graduation gate + run manifest + TensorBoard autostart + run pruning in train.py, critic export (`export_policy.py --value`) for the Unity win-odds bar. All opt-in; existing rungs unchanged.
- **Env throughput without changing results:** removed per-step GPU→CPU syncs (pool pinning, fixed `.any()` checks, mask writes, duplicate frozen loads). `scripts/check_determinism.py` replays the simulator and asserts the new code's sim inputs and every output are bitwise equal to HEAD for r1 and r4league (64 worlds × 200 steps). 256 worlds on the 2060: r1 4,174 → 4,512 policy steps/s, r4league 2,691 → 2,848.
- **One rules source:** `koth/rules.py` holds the hill shape and the out thresholds; `joint_map.json` carries an `arena` block and every `model_dump_*.json` a `robots` list, read by Unity with matching defaults (EditMode test asserts they are equal).

## 2026-10-17/18 — overnight on the RTX 2060 (23:33-08:13): Kim walks and fights, a free-for-all brain

No `training/runs` on this machine, so the shipped brains were rebuilt from their ONNX exports (`scripts/onnx_to_ckpt.py`, actor exact to 4e-6; no critic). Warm starts train a fresh critic first (`--critic-warmup`, actor frozen and learning rate held for 10-20 updates). Unity rules everywhere (no own-leg-contact ending).

| run | iterations, worlds | result | bar | verdict |
|---|---|---|---|---|
| `kim_r0b` (stand, equal-impulse kicks, curriculum) | 600, 4096, 49 min | 97.4 % of impacts survived (1.19 m/s kicks); 58 % at 2 m/s | >= 90 % | pass |
| `kim_r1b` (walk, from r0b) | 1000, 4096, 81 min | impacts 99.5 %; tracking 0.20 m/s / 0.07 rad/s; R3 approach 99.9 % meet within 6 s, 0 self-ejection | 0.15 m/s; R3 >= 90 % | R0 + R3 pass, R1 linear tracking misses (0.20) |
| `ffa1` (r5ffa, 4 G1, warm from careful1, 121 inputs) | 900, 1536, 4 h 04 | vs pool (careful1 x2, gen7, gen1, standing walker): last one in 44.8 %, out 33.6 %, undecided 21.6 %; careful1 in the same seat: 22.4 % / 76.8 % / 0.8 % | last one in >= 40 % | pass; two-fighter regression fails (45 % wins, 42 % undecided: passive in duels) |
| `duel_u1` (r4league fine-tune of careful1, Unity rules) | 400, 2048, 67 min | standing walker 98.2 % (careful1 97.8 %); gen7+gen1 75.5 % (careful1 92.1 %); vs careful1 37 / 33 / 30 % undecided | beat the champion | not promoted |
| `kim_att1` -> `kim_fight1` (Kim pushes a standing Kim off, then a 3-way pool) | 400 + 300, 2048, 79 min | vs standing Kim 99.5 % ejections, median 3.0 s; vs itself 46 / 43 % decided 89 % | r4att >= 70 % | pass |

- **Unity gates.** Kim walker on the flat (`kimr1b`): replay 4e-7, pelvis 1.3 cm, ctrl 5.5 %: pass. Two Kims meeting (`kimr3b`): replay 5e-7, pelvis 2.5 cm, ctrl 11-12 % (bar 10 %). Two Kim fighters (`kimfight`): replay 6e-7, pelvis 5 cm, ctrl 16-23 %: inputs and network verified, closed loop diverges once they touch, the same pattern as the G1 attacker gate (2026-10-08); combat is judged statistically.
- **Promoted:** `Models/kim/walker_policy.onnx` = kim_r1b (was kim_r1 iteration 800); `Models/kim/fighter_policy.onnx` = kim_fight1 (the roster now gives Kim this brain); `Models/ffa_policy.onnx` = ffa1 (new roster entry "G1 Free-for-all"). Critics exported beside each (`*_value.onnx`) for the win-odds bar. The All-rounder stays careful1.
- **Fixed:** `env.py` reported `world_lost` through a view of `self.out` that the reset clears before eval reads it, so FFA evals showed every loss as undecided (training unaffected: `decided` is computed before the reset). Critic exports are checked against a relative bound (values are O(10-100)).
- **Unity free-for-all gate** (`ParityBatch.RunFfaStats`, new): candidate on body A against three opponents drawn per round from the same pool, four G1, no sea, 20 s bell, out = loss, 200 rounds. ffa1: last one in 36 %, out 37 %, undecided 27 %, median win 6.7 s (Python 44.8 / 33.6 / 21.6 %, 6.2 s). careful1 in the same seat: 19 / 81 / 0 % (Python 22.4 / 76.8 / 0.8 %). Unity reproduces the champion closely and confirms the FFA brain about doubles its wins; the remaining gap (more bell ties in Unity) fits the spawn and noise differences (Python spawns at r 0.7-1.2 m with jittered bearings and evaluates with observation noise and DR; Unity spawns at 0.5-1.2 m on even bearings without either).
- **ChatVotes** lived in Hazards.cs, so Unity lost the component after an editor restart (a MonoBehaviour must live in a file of its own name); moved to ChatVotes.cs, with an EditMode test that checks every component's file name.

## 2026-10-18 (day) — Kim duel gate, Kim tracking fine-tune, 5-way play check

- **Kim duel statistics gate** (new `ParityBatch.RunKimDuelStats`, fighter vs itself, no sea, 25 s bell, 200 rounds): Unity 100 % decided (95 / 105), winning time median 6.85 s (quartiles 5.48 / 8.19, mean 7.10). CPU MuJoCo (`KOTH_ROBOT=kim duel_stats.py --policy fighter`): 100 % (109 / 91), median 6.67 s (5.54 / 7.92, mean 6.96). Parity holds. Her testbed predates PolicyRunner's body field, so `BuildDuelDemo` now sets body kinds from the model dump's robots list.
- **`kim_r1c`** (300 iterations from kim_r1b, `tracking_lin_vel` 1.5): tracking 0.114 m/s and 0.073 rad/s (bar 0.15 / 0.2), impacts 99.8 %, R3 99.7 % meet, 0 self-ejection: every bar passes. Unity gate (`kimr1c`, flat walk): replay 7e-7, pelvis 1.9 cm, ctrl 3.3 %. Promoted to `Models/kim/walker_policy.onnx` (walks 2.4 m in 5 s on a 0.6 m/s command, was 2.0 m).
- **5-way play check** (game rules: sea, opportunist targeting, all agents ticked, 8 rounds): Free-for-all 4 wins, Kim 2, All-rounder 1, Duelist 1, Rammer and Walker 0. Results sheet colour dots shrank on long rows (fixed in Hud.uss).
- **`ffa2`** (r5ffa, 600 iterations warm from ffa1, opponent pool now includes ffa1; 2 h 36 on the RTX 2060). Python, 3 seeds x 256 rounds:

  | Run | Pool | Last one in | Out | Undecided |
  |---|---|---|---|---|
  | ffa2 | old (careful1 x2, gen7, gen1, walker) | 59.2 % | 34.5 % | 6.3 % |
  | ffa1 | old (from the overnight run) | 44.8 % | 33.6 % | 21.6 % |
  | ffa2 | new (with ffa1) | 50.4 % | 35.5 % | 14.1 % |
  | ffa1 | new (with ffa1) | 30.3 % | 33.7 % | 35.9 % |

  Two-fighter regression (Unity leg rule): ffa2 ejects the pool opponent 79.2 %, loses 11.8 % (ffa1: 45 % wins, 42 % undecided). The passivity in duels is gone.
- **Unity free-for-all gate, ffa2** (same seat and pool as ffa1's gate, 200 rounds): last one in 47.5 %, out 42 %, undecided 10.5 %, median win 7.5 s (ffa1: 36 / 37 / 27 %). Same Python-to-Unity drop as ffa1 (more bell ties, spawn and noise differences); passes the 40 % bar. **Promoted:** `Models/ffa_policy.onnx` / `ffa_value.onnx` = ffa2.
- **Fixed:** after a statistics gate the director paused the fighters but left `Time.timeScale` 50 and `maximumDeltaTime` 2 s, so an editor run without `-kothExit` kept stepping MuJoCo flat out and stopped answering the CLI (the editor had to be killed). `DemoDirector.StatsDone` now restores real time and leaves play mode. HUD status line wraps instead of overflowing (it pushed the build version off screen); report in `SCREENSHOTS/day2/index.html`.

## 2026-10-18 (evening) — Trump, a third body kind

- **Body** (`Assets/Art/Raw/Trump/Trump.glb`, a rigged GLB with 34 generic bones, arms hanging, 4k textures). No Blender on this machine, so `tools/make_glb_fighter.py` reads the GLB in Python: maps the bones to Kim's names, measures limb thickness from the skin weights, scales to 1.90 m, writes `trump_rig.json` and a 2k-texture GLB that glTFast 6.20 imports. `koth/build_human.py` is now the builder every human shares (Kim's build is byte-identical; `build_kim.py` and `build_trump.py` hold the profiles). Trump: 102 kg, de Leva male fractions, older-man torques (knee 200 Nm, hip 200, ankle 150), wide trunk capsules. Gains first scaled by mass (x1.8 Kim's): the passive hold fell backwards after 5 s; scaled by mass x height (x2.2) it holds with 0.35 deg lean. Unity: the skin follows the bodies within 0.1 mm (SkinFollower now applies the table's scale).
- **Game scene** `koth_6p` (four G1, Kim `e_`, Trump `f_`); Demo_duel is built from it. Kim and Trump now also collide with thrown boxes (Kim never did: her affinity lacked the box bit). Six-colour seat palette. With six fighters the director spawned neighbours 0.5 m apart (less than Trump is wide) and rounds burst apart; neighbours now start at least 0.9 m apart.

| Run | Iterations, envs, time | Result | Bar | Verdict |
|---|---|---|---|---|
| `trump_r0b` (stand, equal-impulse kicks x0.33, curriculum) | 600, 4096, 50 min | 93.6 % impacts survived; 36 % at the G1's 2 m/s (3x his impulse) | >= 90 % | pass |
| `trump_r1b` (walk, from r0b, tracking_lin_vel 1.5) | 1000, 4096, 82 min | tracking 0.107 m/s / 0.075 rad/s, impacts 96.5 %, R3 meet 100 %, 0 self-ejection | 0.15 m/s; R3 >= 90 % | pass |
| `trump_att1` -> `trump_fight1` (r4att then `trump_r4bull`: belly_push 4, arm_contact -1, push_out 1, vmax 0.7) | 400 + 300, 2048, 82 min | vs standing Trump 97.5 % ejections, median 4.5 s; vs itself 43 / 44 %, decided 87 % | r4att >= 70 % | pass, shipped (`Models/trump/fighter_policy.onnx`) |
| `trump_fight2` (from fight1: belly_push 8, arm_contact -4, push_out 0) | 300, 2048, 61 min | vs standing Trump 68 %; loses 62 % to fight1; trunk/arm contact 20 / 19 % (worse) | | rejected |
| `trump_bull1` (from fight1, styled action path with `obs.BULL_POSE`: torso 0.3 rad forward, arms back, knees low) | 400, 2048, 40 min | learned to avoid contact: vs standing Trump 11 % wins, 77 % undecided; loses 85 % to fight1 | | rejected |

- **Style.** New reward terms `belly_push` (opponent's outward speed while one of my trunk capsules touches someone) and `arm_contact`, from new geom kinds 5 (trunk) and 6 (arms) in `_foot_contacts`; `contact_style.py` reports trunk_touch / arm_touch. fight1 in close contact: trunk 33 %, arms 9 % (Kim's fighter 30 / 8 %), so his style is mostly his body: the heaviest fighter, walking at 0.7 m/s, pushing with his trunk. Two attempts at a more distinct style failed (above). The bull pose mechanism (`style_pose` in env/eval/duel_stats/contact_style, `PolicyRunner.stylePose`, roster `Human(..., style)`) is kept, unused until a bull brain passes; next try: a pose blend that ramps in over training instead of switching on at once.
- **Fixed:** a NaN physics state reached PPO through that step's reward (trump_bull1 crashed at iteration 40); rewards of blown-up worlds are zeroed (`env.step`).
- **Unity gates.** Duel statistics (`RunTrumpDuelStats`, fighter vs itself, vmax 0.7, 200 rounds): decided 95 %, median 6.83 s (5.68 / 8.71). Python `duel_stats.py --vmax 0.7`: 95 %, 6.90 s (5.68 / 8.83): pass. Six-way play (game rules, 8 rounds): Free-for-all 6, All-rounder 1, Duelist 1, Trump and Kim 0 (Trump second, out at 12.6 s, in round 1). EditMode 31/31, PlayMode 2/2.

## 2026-10-18/19 (night) — bull charge eased in, G1 free-for-all generation 3, cross-body training setup

| Run | Iterations, envs, time | Result | Bar | Verdict |
|---|---|---|---|---|
| `trump_bull2` (trump_bull with `style_ramp_iters` 300: pose and smoothing ease in from fight1's stance) | 600, 2048, 81 min | vs standing Trump 13 % wins, 71 % undecided; vs fight1 6 % wins | r4att >= 70 % | rejected: like bull1 it learned to stay out of contact. The bull pose is dropped (code kept, unused). |
| `ffa3` (r5ffa, warm from ffa2, pool ffa2 x2, ffa1, careful1, gen7, standing walker) | 500, 1536, 2 h 37 | vs ffa2's pool 51.3 % (ffa2 50.4 %); vs pool3 35.7 % last one in (ffa2 33.9 %); two-fighter 87.8 % wins (ffa2 79.2 %) | | promoted |

- **Unity free-for-all gate under the new spawn spacing** (neighbours >= 0.9 m apart; ffa2's earlier 47.5 % used 0.5 m): ffa2 51 / 42 / 7 %, median win 8.1 s; ffa3 51.5 / 40.5 / 8 %, median 7.2 s. Equal in four-way play, better in duels: `Models/ffa_policy.onnx` = ffa3.
- **Cross-body training.** Every brain so far fought only its own body; in mixed scenes Trump's fighter beats the G1 league (careful1 x2, gen7, gen1) 31 % and loses 61 %, Kim's 34 / 60 %. `python -m koth.build_mjcf mix <body> <body> ...` builds mixed scenes (G1 base, humans swapped in at the same place and collision bit) into assets/g1; the env reads each robot's body from the scene's dump and keeps default pose, joint and control ranges, standing height, foot height and style offsets per robot (same-body scenes unchanged). Rungs `<body>_vs_<other>` (duel, frozen opponent of another body) and `<body>_ffa` (a human against three G1, 121 inputs). Unity: `RunFfaStats("Trump"|"Kim")` runs the free-for-all gate in koth_6p; the roster takes a human's newest brain (ffa, xg1, fighter, walker; `<file>_style.txt` names a styled pose).
- **Cross-body duels, first two tries (both rejected).** Eval against the G1 league (careful1 x2, gen7, gen1, deterministic), win / loss / undecided; starting brains: Trump fight1 32 / 59 / 9 %, Kim fight1 32 / 63 / 5 %.

  | Run | Recipe | Trump | Kim |
  |---|---|---|---|
  | `xg1` (400 it) + `ffa1` (300 it, a human vs three G1) | G1 opponents stochastic (rebuilt actors' std 0.3), no critic warm-up | 10 / 69 / 21; four-way 11 % last one in (fight1 17 %) | 19 / 80 / 1; four-way 4 % (fight1 14 %) |
  | `xg2` (300 it) | deterministic G1 (`train.py --frozen-deterministic`), critic warm-up 20 | 6 / 29 / 65 (stalls) | 20 / 53 / 27 |

  Why xg1 failed: the rebuilt G1 actors acting with std 0.3 mostly fall on their own (fight1 beats them 84 %, the deterministic ones 29 %), so the humans learned to wait; training reported wins 2.4x losses while eval showed the reverse. xg2 stalled because a draw costs the same as a loss (-10) and a long stall against a stronger opponent avoids both the loss and the effort. Next: v3 (scripts/queue_xb3.sh), a standing-G1 attacker stage first (rung `<body>_att_g1`), then the league with draws at -20.
- **Cross-body v3 (rejected).** Standing-G1 attacker stage (`<body>_att_g1`, 250 it, deterministic, critic warm-up 20, draw -20), then the league (`<body>_vs_g1`, 300 it, G1 league + standing walker, draw -20). The attacker stage works: Trump pushes a standing G1 off 97.5 %, Kim 88 %. The league stage again ends below the start: vs the G1 league Trump 21 / 71 / 8 %, Kim 15 / 76 / 8 % (both fight1 32 %); vs a standing own body Trump 45 % (fight1 97.5 %), Kim 70 %. Three recipes, six runs, the same drift: against the deterministic G1 league a win is rare, and every league stage settles below where it started. Cross-body training is parked; the humans keep fight1. Ideas for later: keep own-body opponents in the pool so wins stay common, or train the G1 side against the humans instead.

## 2026-10-19 — broadcast features: get-up brain, Tactician A/B gate (RTX 2060 machine)

Seven features from the feature list (tale of the tape, vitals HUD, knock-out credit + feed + voice, Tactician, pack
targeting, get-up rule, stamina) went into the game; two needed training or a gate.

**Get-up brain (rung `getup`, STAGED_TRAINING.md section 7).** One robot on the flat floor, 85 % of episodes start from a
0.85 m drop in a random orientation, nothing but a NaN ends an episode; 4096 envs, about 3.8 s per iteration on the 2060
(8 s while the Unity editor plays a six-way match: leave play mode during training).

| Run | Recipe | Result | Verdict |
|---|---|---|---|
| `getup1` | height 2.0, upright^2 1.0, standing pose 1.0, drift -0.2; 2000 it planned, stopped at 900 | settled by iteration 400 into sitting: from drops, pelvis at 26 % of the standing height, up-axis 0.65, standing 1 % (eval without noise, 256 drops, 8 s) | rejected |
| `getup2` | warm from getup1 900; pelvis pull 0.8 x body weight while low, fading over 900 it; + upward-speed reward 0.5 | standing share stayed 16 % (the standing starts) up to iteration 450 with the pull still 0.4-0.8: the sitting habit (action std 0.25) did not move | rejected, stopped at 450 |
| `getup3` | fresh; pull 1.0 x body weight at any height, fading to 0 over iterations 0-1000 (HoST-style), 1600 it, 115 min | standing share 5 % (it 50) -> 53 % (200) -> 86 % (450, pull 0.55) -> 84 % (800, pull 0.2) -> 88 % (1600, no pull; the 15 % standing starts and the first ~1.5 s of every drop count as not standing). Unaided eval (256 drops, DR on): at 600 4 % (leaning on the pull), at 1000 85 % standing by 2 s / 100 % by 4 s, final 98 % by 2 s / 100 % by 4 s. Graduated: last-100 mean reward 28.7 (bar 18), 0 NaN updates | **shipped** `Models/getup_policy.onnx` |

- **Get-up gates (final brain).** Drops, 200, training recipe: CPU MuJoCo `getup_stats.py` 100 % up within the 5 s count, median 1.52 s (1.40 / 1.60); Unity `RunGetupStats` 99.5 % (one drop never stood in 8 s), 1.47 s (1.38 / 1.60): pass. The game rule in duels (champion vs itself, 200 rounds, `--getup` / `RunGetupDuelStats`): Python 99.5 % decided (105 / 94 / 1), median 8.40 s (5.71 / 12.17); Unity 99 % (103 / 94 / 3), 7.98 s (5.75 / 11.81): pass, the same Unity-ends-sooner offset as the plain-rule gate (8.05 vs 8.40 s). Get-ups are rare in duels (most rounds end on a ring-out): the rule changes the Python median by 0.0-0.14 s. Kim and Trump have no get-up brain yet (same rung with `KOTH_ROBOT`). PlayMode 3/3 (new: intro into the first round).

**Tactician A/B gate** (`ParityBatch.RunTacticianStats`: Gate_duel_2p, champion vs champion, 200 rounds, 25 s bell, seat
A plans with rollouts on 4 worker threads and the physics waits for them; wins A - B):

| Variant | A - B (ties) | Press chosen |
|---|---|---|
| Press only (sanity: the hook path) | 97 - 100 (3) | 100 % |
| 0.6 s horizon, no prior | 27 - 171 (2), again 27 - 173 | 39 % |
| 1.2 s, Press prior 0.5 | 64 - 133 (3) | |
| 2.0 s, prior 0.3 | 65 - 128 (7) | 59 % |
| 1.2 s, prior 1.0 (now the default) | 76 - 120 (4) | 87 % |
| 1.2 s, prior 0.3, 3 averaged rollouts with +-0.05 m/s nudges | 74 - 125 (1) | 69 % |

Every planning variant loses to plain Press, even when it deviates in only 13 % of decisions. The brains were trained on
one goal law (straight at the opponent), and contact fights diverge too fast for 0.6-2 s rollouts to tell a good
deviation from a lucky one. The Tactician stays as an experimental show option (off by default, "Tactician
(experimental)"); not a gameplay improvement. A real tactical layer would need brains trained with varied goal points.

**Other measurements.** MlpNet (C#, worker threads) vs Sentis on the same ONNX: max error below 2e-4 x output scale on
four brains (EditMode `FeatureTests`). New components cost nothing measurable on the physics tick: six-way match, editor,
tick 1.0-1.15 ms with ContactLedger, Biometrics, Stamina on or off. Stamina keeps the servo scale inside 0.85-1.15
(trained kp randomisation +-20 %); off in every gate. EditMode tests 40/40.

## 2026-10-19 (night) — graphics and audio pass, self-collision audit

Nine of ten GFX & Audio options (the tenth, a Blender MCP summit upgrade, waits for Blender on this machine). All render or
audio only: no observation, action, ctrl or model change, and every new component is inert in the statistics gates.

- **Duel statistics gate re-run** (`RunDuelStats`, champion vs itself, 200 rounds, gate scene rebuilt with every new
  component): decided 100 % (94 / 106 / 0), median 7.70 s (5.66 / 10.70). Previous Unity run 99 %, 8.05 s (5.07 / 11.08);
  Python 98.5 %, 8.40 s (5.64 / 12.16): pass, the median moved by about one standard error (~0.35 s for 200 rounds) and
  keeps the known Unity-ends-sooner offset. `duel_stats_unity.json` updated. EditMode 43/43 (new `CollisionTests` x3),
  PlayMode 3/3.
- **Self-collision audit** (`training/scripts/collision_audit.py`, report `SCREENSHOTS/collision-261019/`, Unity mirror
  `EditorTools.CollisionAudit` + `CollisionTests`, which find the same pairs and depths). Fighter `a_` of each
  `scene_koth_2p`: 32 self-pairs collide (foot / shin against the other leg); everything else is off by design, against
  AGENTS.md section 3. G1 and Trump: no overlap on a colliding pair in zero, rest or T-pose. **Kim: her foot boxes
  interpenetrate 3.5 mm (and two 0.5 mm foot pairs) in her rest pose**, so every Kim episode starts with her feet pressing on
  each other; fixing her stance or foot width changes her physics and waits for her next retrain (the test allows exactly
  these three pairs at their depth). With every non-adjacent pair on, the humans' pelvis capsule would overlap the thighs by
  136 mm (Kim) / 166 mm (Trump) at rest: turning AGENTS.md's rule on needs excludes for the dummy hip chain first.
- **Audio levels** (editor, six-way fights, Master Mid / Ambience Mid, RMS after the mix): wind 0.008-0.025, surf
  0.004-0.033, crowd 0.017-0.062. The first crowd setting was 6x louder (RMS up to 0.51 at a round's end) and clipped the
  filter at 1.0; after the retune its peak stays at 0.44-0.52. 5 six-way rounds: 237 thuds, 143 clanks, 2 smacks, 1 grunt,
  387 3D one-shots, 9 slow-motions, 5 hit stops.
- **Player performance** (release, 540x960, `-kothPerf 30`, six-way, everything on, quality High): 54.5 / 59.1 FPS in two
  runs, frame p50 16.7 ms, p99 26-27 ms (one 2.1 s launch hitch in run 1 only); physics tick 1.0 ms mean (p99 2.9) for six
  bodies, 459-477 of 500 steps/s. In the editor the frame rate is about 40 FPS whatever the quality preset or Effects
  switch: the six-body physics, not rendering, bounds it there.
- **Rejected looks on the way** (captures in `SCREENSHOTS/gfx-261019`): danger rim from 35 cm inside the rim lit almost
  every fighter red (now 12 cm, as the danger light); 4 mm scratches aliased into glitter (now 9 mm with a pixel-width fade);
  hit flash from contact force > 150 N stayed on while fighters leaned on each other (now only on `MatchEvents.Impact`);
  a 0.9 m bolt looked like a cartoon at 40 m (now ~2 px at any distance); the first shockwave was invisible at phone size, the
  second read as a fog band.
