# Staged training (prepared 2026-10-17; stages 2, 3 and 6 trained 2026-10-18, see rl_optimization_log.md)

Everything below is opt-in. Existing rungs (r0 r1 r2 r3 r4 r4att r4sp r4league r4shove r4probe) produce the same
configuration as before and, verified bit for bit, the same environment trajectories (see "Throughput" below).
No checkpoints exist on this machine (`training/runs` is absent); the commands assume the league checkpoints from
the training machine are copied back into `training/runs/league/` (`allround4.pt` = current champion,
`attacker_policy.onnx`; `gen1..7.pt`, `careful1.pt`, `walker.pt`) and `runs/r1/r1_b/model_final.pt`.

All commands run from `training/`. Add `--tensorboard` to any `train.py` call to start `tensorboard --logdir runs`
(port 6006). Every run now writes `runs/<rung>/<name>/manifest.json` (cfg, git commit, seed, args, bars,
start/end time). `--graduate-reward R` exports to Unity only when the last-100-episode mean reward is >= R and
no update had a NaN gradient; no target value is known for the new rungs, so leave it off for the first run of
each and set it from that run's `Train/mean_reward` once its eval passes. Prune afterwards with
`uv run python scripts/prune_runs.py` (dry run) / `--delete`.

## Order

| # | what | why first |
|---|---|---|
| 1 | measure the rules gap (no training) | decides whether step 4 is needed |
| 2 | Kim recipe b: `kim_r0b` -> `kim_r1b` | she has no brain that meets any bar; 103 inputs, no Unity change |
| 3 | FFA brain `r5ffa` (121 inputs) | the game is a 4-5 way free-for-all; needs the Unity obs mirror |
| 4 | rules-matched fine-tune of the champion (leg contact off) | only if step 1 shows a gap |
| 5 | experiments: anti-stalemate, decision rate, fatigue | each changes play or needs a Unity mirror; one at a time |
| 6 | value net export for the win bar | any finished brain |

## 1. Rules gap (eval only)

Training and `eval.py` end a round on own foot-shin contact; Unity and `duel_stats.py` do not.

```bash
uv run python scripts/eval.py --rung r4league --ckpt runs/league/allround4.pt --opponent runs/league/allround4.pt --seeds 3 --num-envs 256 --seconds 25
uv run python scripts/eval.py --rung r4league --ckpt runs/league/allround4.pt --opponent runs/league/allround4.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
```

If win/loss/tie move by more than ~3 points between the two, do step 4. The new rungs (`r5ffa`, `kim_r0b`,
`kim_r1b`) already train with `terminate_on_leg_contact = False`. Unity mirror: none (Unity is the reference).

## 2. Kim recipe b (`KOTH_ROBOT=kim`)

`kim_r0b` / `kim_r1b` = r0 / r1 plus `kim_overrides()` in `koth/env.py`:

| change | value | reason |
|---|---|---|
| kick size | x m_G1 / m_Kim = 33.34 / 55.99 = 0.595 (0.30-1.19 m/s) | a kick is a velocity change; this keeps the G1's impulse. Boxes are bodies and already carry the right momentum |
| disturbance curriculum | kicks and box speeds 25 % -> 100 % over 300 iterations (`push.curriculum_iters`) | `kim_r0` never kept a robot up for 20 s; continues correctly on `--resume` |
| `swing_height` | 0.09 m (G1 0.12) | her thigh is 0.27 m and her hips sit low |
| pose penalties | pose -0.05, hip -0.15, knee -0.05 (halved) | about 2/3 of the G1's torque per kg (knee 150 Nm / 56 kg vs 139 / 33), she needs larger excursions to recover |
| leg-contact termination | off | short thighs make foot-shin touches common; Unity does not end on them |

Masses are read from the model dumps (`koth.env.robot_mass`).

```bash
export KOTH_ROBOT=kim
uv run python scripts/train.py --rung kim_r0b --name kim_r0b --num-envs 4096 --iters 600 --tensorboard
uv run python scripts/eval.py  --rung kim_r0b --ckpt runs/kim_r0b/kim_r0b/model_final.pt --seeds 3 --num-envs 256           # kicks 1.19 m/s
uv run python scripts/eval.py  --rung kim_r0b --ckpt runs/kim_r0b/kim_r0b/model_final.pt --seeds 3 --num-envs 256 --push-vel 2.0
uv run python scripts/train.py --rung kim_r1b --name kim_r1b --num-envs 4096 --iters 1000 --resume runs/kim_r0b/kim_r0b/model_final.pt
uv run python scripts/eval.py  --rung kim_r1b --ckpt runs/kim_r1b/kim_r1b/model_final.pt --seeds 3 --num-envs 256
uv run python scripts/eval.py  --rung r3      --ckpt runs/kim_r1b/kim_r1b/model_final.pt --seeds 3 --num-envs 256 --leg-contact-rule unity
uv run python scripts/export_policy.py --ckpt runs/kim_r1b/kim_r1b/model_final.pt --name walker                                # -> Models/kim/
```

Bars: R0 >= 90 % of impacts survived (equal-impulse kicks; report the 2 m/s figure too), R1 tracking error
< 0.15 m/s and < 0.2 rad/s, R3 >= 90 % of pairs meet within 6 s with <= 2 % self-ejection (current brain: 40-43 %).
Unity mirror: none (103 inputs, 50 Hz, same rules). Gate: `ParityBatch.RunPolicy -kothRobot kim -kothScene kim_koth_2p`
after `record_reference.py` with `KOTH_ROBOT=kim`.

## 3. Free-for-all brain `r5ffa`

Four G1 on `scene_koth_4p_train.xml` (already generated, unchanged). Learner `a_`, frozen opponents on `b_ c_ d_`,
each opponent drawn per world slot from the pool. "The opponent" of every robot is its nearest opponent still in
the round (eliminated robots are skipped; they stay in the world limp, ctrl 0, like Unity's paused runner).
The learner's observation gets a second 9-value combat block for the second-nearest opponent still in (the
nearest again when only one is left): 121 inputs. Frozen 112-input opponents read the first 112 columns, exactly
what Unity gives today's brains. Episode: learner out = loss, all three opponents out = win (`win` +30),
20 s bell = draw (-10); leg-contact termination off.

```bash
POOL="runs/league/allround4.pt runs/league/gen7.pt runs/league/careful1.pt runs/league/allround1.pt runs/league/walker.pt:stand"
uv run python scripts/train.py --rung r5ffa --name ffa1 --num-envs 2048 --iters 600 --warm runs/league/allround4.pt --opponent $POOL --tensorboard
uv run python scripts/eval.py  --rung r5ffa --ckpt runs/r5ffa/ffa1/model_final.pt --opponent $POOL --seeds 3 --num-envs 256 --seconds 20
uv run python scripts/eval.py  --rung r4league --combat2 --ckpt runs/r5ffa/ffa1/model_final.pt --opponent runs/league/gen3.pt runs/league/gen4.pt runs/league/gen5.pt runs/league/gen6.pt runs/league/gen7.pt --seeds 3 --num-envs 256 --seconds 25 --leg-contact-rule unity
uv run python scripts/export_policy.py --ckpt runs/r5ffa/ffa1/model_final.pt --name ffa
```

`--warm` pads 112 -> 121 actor inputs (new weights zero: it starts as allround4) and 121 -> 130 critic inputs.
Bars: last one in >= 40 % of rounds against the pool (25 % = no better than the opponents); in two-fighter
regression it should keep allround4's bar (share of decided > 55 %, ties < 15 %).

Unity mirror (C# side, not done here):
- `ObsBuilder.FillCombat` a second time at offset 112 with the second-nearest fighter still in (nearest again if
  only one is left), same 9 formulas; the first block stays the nearest fighter still in (as `DemoDirector` does).
- `PolicyRunner` obs dim 121 (read it from the ONNX input shape) and fill both blocks for 121-input brains; in a
  two-fighter round both blocks describe the one opponent.
- Rules: eliminated fighters paused (already), no leg-contact ending (already).

## 4. Rules-matched fine-tune (only if step 1 shows a gap)

```bash
uv run python scripts/train.py --rung r4league --name allround_u1 --num-envs 3072 --iters 350 --warm runs/league/allround4.pt \
  --opponent runs/league/walker.pt:stand runs/league/walker.pt:stand runs/league/gen7.pt runs/league/allround4.pt --reward self_edge=-5.0 --leg-contact-rule unity
bash scripts/judge.sh allround_u1    # after copying model_final.pt to runs/league/allround_u1.pt
```

## 5. Experiments (one variable each, against the same pool as the run they modify)

| option | how | default | Unity mirror |
|---|---|---|---|
| anti-stalemate | `--reward stalemate=-1.0`: penalty per second once both robots are within 0.6 m and both pelvis speeds < 0.1 m/s for > 2 s (`cfg["stalemate_rule"]`) | off (0.0) | none (reward only) |
| push_out cap | `--push-out-cap 2.0`: at most 2.0 push_out reward per episode | off (0.0) | none |
| decision rate | `--decimation 20` (25 Hz) / `15` (33 Hz); `ctrl_dt = decimation x 2 ms`; eval with the same `--decimation` | 10 (50 Hz) | `PolicyRunner` must infer every `decimation` physics steps and advance the gait phase by 2 pi 1.5 ctrl_dt; today Unity reads one global `decimation`/`ctrl_dt` from joint_map.json, so a per-brain rate needs a per-brain setting |
| fatigue | `--fatigue on` (or `obs`): per-joint state F in [0,1], dF/dt = (2 P - F) / 10 s with P = abs(tau qdot) / (tau_max 10 rad/s); force range x (1 - 0.4 F) | off | `on`: Unity must scale each actuator's forcerange the same way (else the brain meets stronger joints than it trained with: safe but unmatched); `obs`: also append the 29 F values after the combat blocks (132/141/150 inputs) |

Each is judged by `judge.sh` / `eval.py` with the same switches it was trained with.

## 6. Value net (win bar)

```bash
uv run python scripts/export_policy.py --ckpt runs/league/allround4.pt --name attacker --value     # + attacker_value.onnx
```

Input = critic observation (121 for a 112 brain): 103 noise-free locomotion obs, world pelvis linear velocity (3),
pelvis height above ground (1), feet contact flags (2), DR params (mass scale 1, friction, kp scale 1; use 1, 0.6,
1 in Unity), then the 9 combat inputs. Output `value` (1,1) = expected discounted return, not a probability:
use `sigmoid((V_a - V_b) / 10)` or similar as a bar. Checked on a synthetic rsl_rl checkpoint: ONNX vs torch max
error 2.2e-6 (policy 2.9e-6), opset 17, batch 1.

## 7. Get-up brain (rung `getup`, added 2026-10-19 for the game's get-up rule)

One brain per body kind that takes over while a fighter is down (`DemoDirector.OutWithGetUp`, `PolicyRunner.SetGetUp`):
103 inputs, zero command, `ctrl = default + 0.5 action`, no style path. Flat floor, one robot.

| setting | value | reason |
|---|---|---|
| starts | 85 % drops from 0.85 m in a uniformly random orientation, joints default +- 0.6 rad; 15 % standing | it must get up from anything a fight ends in, and stay up once up |
| ending | only a NaN state; 8 s episodes | lying down is the task, not a failure |
| rewards (per s) | height share 2.0, uprightness^2 1.0, once standing (h > 0.85 x standing, up > 0.9) default pose 1.0 and no drift -0.2, upward speed 0.5 until standing; action rate -0.01, joint speed -2e-4, joint limits -1 | dense progress toward standing, then a familiar stance for the fight brain |
| assist | vertical pull on the pelvis, 1.0 x body weight, fading to 0 over iterations 0-1000 (`assist`, `assist_iters`, `assist_low_only`) | without it the policy settles into sitting (getup1); a pull applied only while low, from a sitting policy, did not move it (getup2). After HoST (2025) |
| disturbances | no kicks, no boxes; DR as every rung | |

Gates: `scripts/getup_stats.py` (CPU, 200 drops: success within the game's 5 s count, time-to-stand quartiles) vs
`ParityBatch.RunGetupStats` (Unity, Testbed_flat_1p, `GetupProbe`); then the game rule: `duel_stats.py --getup` vs
`ParityBatch.RunGetupDuelStats`. Runs and numbers: rl_optimization_log.md (2026-10-19).

## 8. Calibration, action path, rewards and curriculum (added 2026-10-20)

All opt-in; every existing rung is bitwise unchanged (`check_determinism.py`, r0 r1 r4league r4sp r5ffa getup).

| option | how | Unity mirror |
|---|---|---|
| action map | `--action-map tanh`: a = tanh(raw) instead of clamp(raw, -1, 1) (`koth/action_map.py`) | `ActionMap.cs` via `<name>_action.json` |
| per-joint scale | `--action-scale runs/audits/scale_<brain>.json` from `scripts/action_scale.py` (p95 of the audited brain's raw outputs x 0.5, twins equal, capped by the joint's reach). With `--warm`, the change is folded into the actor (output rows, std, last-action normaliser, exact to 3e-6 rad) | same sidecar |
| torque window | `--torque-window`: target clamped to q + (kv qd -+ tau) / kp, nominal gains | same sidecar |
| reward clip | `--no-reward-clip`; `rew/clip_share` and `rew/clipped_away` are logged on every run | none (reward only) |
| potential shaping | rung `r4pbrs`: ring_advantage / opp_radius / self_edge / push_out off, `potential` 10 (F = 0.99 phi' - phi, phi = (opp radius - my radius) / 1.7, outside the clip, 0 at terminal states) | none |
| tactics | rung `r4tactic`: learner goal point drawn from `obs.TACTICS` (Press 50 %, Flank L/R, Regroup, Lurk 12.5 % each), held 1-3 s; tracking 0.6 / heading 0.3 | `Tactics.Goal` (same law); inputs unchanged (112) |
| curriculum | `--curriculum ladder[:bar:demote]` with `--opponent` weakest first: stage i draws mostly opponent i, last stage all evenly; advance at win share >= bar (0.5) over 20 iterations after >= 40 in the stage, step back below demote (0.3) (`koth/curriculum.py`) | none |

Every finished run writes `graduation.json`; `export_policy.py` refuses a run with NaN updates or a missed
`--graduate-reward` target unless `--force`, and writes or removes `<name>_action.json`. The CPU reference
(`duel_stats.py`) reads the sidecar too. Queue: `scripts/queue_options.sh` (G1 runs repeat duel_u1's recipe and change
one thing), then `scripts/queue_bodies.sh` (Kim's feet, Trump's hold). Results: rl_optimization_log.md 2026-10-20.

## New JSON keys for the Unity side (read with fallbacks)

- `joint_map.json` (`training/assets/{g1,kim}/`, `Assets/PoKingHill/Models/`, `Models/kim/`), last key:
  `"arena": {"plateau_r": 1.5, "slope_k": 0.1111111111111111, "out_radius": 1.7, "fall_height": 0.3, "fall_up": 0.3}`
  (= `koth/rules.py`; Unity's out test is radius > out_radius, pelvis up-axis z < fall_up, pelvis height above
  ground < fall_height, ground z = -slope_k max(r - plateau_r, 0)^2).
- every `model_dump_<tag>.json`, last key: `"robots": [{"prefix": "a_", "body": "g1"}, ...]` in model order
  (koth_5p: a_..d_ g1, e_ kim; Kim's own scenes: body kim). Written after the train/unity dump-equality check.
- `koth/build_mjcf.py` now writes both keys, so a rebuild reproduces these files. A rebuild drops the stale
  `robot_prefixes` key that the committed joint_map.json files still carry (its writer was removed in 38a112e).

## Tools added

- `scripts/obs_audit.py --ckpt <ckpt> --tag <scene tag>`: Unity observations (`assets/<robot>/unity_obs_<tag>.jsonl`,
  one array per line, to be written by the Unity side) vs the checkpoint's normalizer and vs Warp samples: per input
  z mean/std, share beyond |z| 4, mean drift (training sd) and two-sample KS distance.
- `scripts/drive_audit.py [--ckpt <ckpt> --rung <rung>]`: kp/kv/force range per joint vs the Unitree G1 table
  (Kim: her design table) and, with a checkpoint, the share of actions on the +-1 clamp, of ticks at the torque
  limit, and peak joint speeds. The G1 model matches the published peak torques on all 29 joints; leg kp is
  2-3x unitree_rl_gym's by design (log 2026-10-07).
  2026-10-20: `--json` rows also carry raw-output p50/p95 (before the clamp); `scripts/audit_all.sh` audits every
  shipped brain into `runs/audits/drive_<name>.json`.
- `scripts/step_response.py`: per joint and body, step response with the pelvis held (rise, overshoot, settling,
  steady-state error, torque limit), effective inertia, natural frequency and damping ratio, and a left/right torque
  comparison of the standing hold.
- `scripts/reward_audit.py` (`reward_audit_all.sh`): per-term reward shares on the training cfg, how often the zero
  clip binds and how much penalty it erases, dominance, stall (bell vs win) and farm flags.
- `eval.py` duel / free-for-all rows carry `contact_share` and `flags` (stall, contact_avoidance, self_ejection).
- `ParityBatch.RunObsDump(robot, rounds, a, b, ffaCandidate, tag, brain)` records Unity's inputs (`ObsRecorder`) for
  `obs_audit.py --unity ... [--brain <onnx name>]`.
- `scripts/check_determinism.py`: reference (HEAD) vs working-tree env/eval, bitwise, plus throughput.
- `scripts/prune_runs.py`: dry-run-first pruning of short (< 5,000 env steps), checkpoint-less or abandoned runs.

## Throughput (task A) and verification

Changes (all in `koth/env.py`, `koth/obs.py`, `scripts/eval.py`): box pinning and parking through precomputed
pool column slices and `torch.where` (no per-box `int()` syncs, no boolean-mask writes); `FrozenPool` stander /
shove flags as Python bools; projectile block skipped when projectiles are off; `_foot_contacts` scatters to a
dummy slot instead of masking; `_rewards` no longer builds the full 103-dim observation for 6 columns, gravity is
computed once per step and shared with the tipped test, the down vector is built by fill kernels (no host copy),
separation and feet positions are computed once; each opponent checkpoint is loaded once and shared; `eval.py`
accumulates on the GPU and reads back once per seed.

Verification (`scripts/check_determinism.py --rungs r1 r4league --num-envs 64 --steps 200 --eval --bench 256`,
RTX 2060, reference = HEAD). mujoco_warp is not run-to-run deterministic (the same code differs by ~1e-6 after one
step), so the reference run records every `_step_sim`'s inputs and outputs and the new run replays them, asserting
its own sim inputs are bitwise equal. Result: all 10 old rungs give the same cfg; r1 (frequent kicks and boxes,
135 resets) and r4league (stochastic frozen pool with a shared module, `:stand` and `:shove` entries, 315 resets)
are bitwise identical in sim inputs, qpos, qvel, policy and critic observations, rewards, every reward term, dones
and time-outs over 200 steps; eval.py's metric rows are identical (eval_flat, eval_duel, 2 seeds x 500 steps). The
harness does catch differences: a reward scale of 1 + 1e-7 or swapping two random draws fails it.

| 256 worlds, random-actor rollout (many resets) | HEAD | now |
|---|---|---|
| r1 | 4,174 policy steps/s | 4,512 (+8.1 %) |
| r4league (frozen pool) | 2,691 | 2,848 (+5.8 %) |

Zero-action throughput and GPU memory on the RTX 2060 (`env.step` only, no PPO), for sizing: r1 8,970 steps/s at
256 worlds (28.5 ms/step, +269 MB); r4league 3,299 learner steps/s (77.6 ms, +469 MB); r5ffa 1,754 learner
steps/s (146 ms, +769 MB). Memory grows by about 0.5 / 1.2 / 2.2 MB per world for 1 / 2 / 4 robots, so on 6 GB:
r0/r1 4096 worlds fit (~2.1 GB), two-robot rungs at most ~3072 (~3.9 GB; 4096 needs ~5 GB), r5ffa at most ~1536
(~3.5 GB). Larger world counts were not run on this machine.

Not changed on purpose: `_step_sim`'s `torch.cuda.synchronize` + `wp.synchronize_device` around the graph launch
(torch and Warp use different streams; replacing the host syncs with stream events is the next win but risky).
Other per-step host syncs that remain because removing them would change the RNG stream or the batch shapes:
`push.any()` / `proj.any()` and their `nonzero()` (random draws sized by the number of events), `nacon.item()`
in `_foot_contacts`, `FrozenPool`'s per-net `sel.any()` and boolean-mask gathers, and the reset `nonzero()`.

## GPU time estimates

5070 Ti figures come from the log (flat single robot about 37k policy steps/s at 4096 worlds; the league trained
350 iterations at 3072 worlds in about 26-30 min, `allround_league.py` budgets 5.5 s per iteration). The RTX 2060
column is an estimate, not a measurement: about 3x slower per iteration (a third of the memory bandwidth and
compute) at the world counts that fit in 6 GB. Measure `python -m koth.env <rung> <N>` at the chosen N before
planning.

| run | worlds (5070 Ti / 2060) | iterations | RTX 5070 Ti | RTX 2060 (est.) |
|---|---|---|---|---|
| kim_r0b | 4096 / 4096 | 600 | ~35 min | ~1.5-2 h |
| kim_r1b | 4096 / 4096 | 1000 | ~1 h | ~2.5-3.5 h |
| r5ffa | 2048 / 1536 | 600 | ~1 h (4 robots per world) | ~3-4 h |
| r4league fine-tune (step 4) | 3072 / 2048 | 350 | ~30 min | ~1.5-2 h |
| each step-5 experiment | as its base rung | 350 | ~30 min | ~1.5-2 h |
| evals (3 seeds x 256) | 256 | - | ~1-2 min each | ~3-5 min each |

One training at a time: two 4096-world trainings plus an eval already filled the 12 GB card.
