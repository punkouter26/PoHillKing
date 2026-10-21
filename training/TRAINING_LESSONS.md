# Training lessons (G1, Kim, Trump; 2026-10-07 to 2026-10-20)

What three bodies taught us about getting a physics-simulated fighter to stand, walk and push, written for the next
custom rigged mesh. Every claim points at the run or measurement in `DOCS/rl_optimization_log.md` that showed it. The G1
left the game on 2026-10-20; its training code, checkpoints (`training/runs/`, `runs/league/`) and assets
(`training/assets/g1`) stay, because the human recipes below were derived from it and the mixed-body scenes are built
on it.

## 1. A new body, step by step

| step | how | bar / check | lesson behind it |
|---|---|---|---|
| rig to profile | `tools/make_glb_fighter.py <body>` (rigged GLB, no Blender) or a Blender script like `make_kim.py`; then `koth/build_<body>.py` with mass, de Leva segment fractions, peak joint torques, human ranges, `DEFAULT_POSE` | the G1's 29 joint names and order, so networks, env and Unity runner are shared | Kim and Trump reuse every line of training and runtime code because of this |
| gains | start from Kim's `PD_GAINS` scaled by **mass x height** (not mass) | 10 s passive hold in Warp and CPU MuJoCo, then Unity `ModelDumpCheck` 0 mismatches | Trump's mass-only scaling (x1.8) fell backwards after 5 s; mass x height (x2.2) holds with 0.35 deg lean |
| feet | `build_human.py` keeps a 4 mm gap between the feet (`FOOT_GAP`) | `collision_audit.py`: 0 overlapping enabled pairs at zero, rest, T-pose | Kim's narrow stance started every episode with her foot boxes 3.6 mm into each other; fixed 2026-10-20, her brains measured the same before and after |
| hold ctrl | `build_mjcf.symmetric_hold` mirror-averages the stance targets when a settle comes out asymmetric | `step_response.py`: left / right hold torques equal | a double stance is statically indeterminate: Trump settled leaning (ankle roll -2.6 / -0.6 Nm); brains were unaffected (targets are default + action) |
| drive check | `step_response.py`, `drive_audit.py` | damping ratio per joint, torque limit reached in a step | every body so far: hips and waist zeta 0.12-0.22, ankles and wrists 1.4-2.5. Brains learn around it; changing gains means retraining |
| stand | rung `<body>_r0b`: equal-**impulse** kicks (2 m/s x m_G1 / m_body), disturbance curriculum 25 % -> 100 % over 300 iterations, halved pose penalties, own-leg-contact ending off | >= 90 % of impacts survived | Kim's first r0 with the G1's settings: no robot stayed up 20 s (log 2026-10-09) |
| walk | `<body>_r1b` from r0b, then a short `tracking_lin_vel 1.5` fine-tune if tracking misses | 0.15 m/s / 0.2 rad/s tracking; R3 approach >= 90 % | Kim r1b tracked 0.20 m/s; r1c (300 iterations, tracking 1.5) 0.11 |
| push | `r4att`: push a **standing** copy of yourself off the rim | >= 70 % ejections | a defender that walks back to the centre cannot be shoved out by an equal body (r4att_a/b: 0 wins); one standing near the rim can (98.7 %) |
| fight | `r4league` / `<body>_r4bull`: a pool of frozen earlier brains, standing opponents kept in it | win share of decided > 55 %, ties < 15 % | without standing opponents the league forgets how to beat one (gen6: 4-7 %) |
| get up | rung `getup` with the HoST-style pull: 1.0 x body weight at any height, fading to 0 over 1000 iterations | 100 % up within the 5 s count | without the pull, or with a weaker low-only pull from a sitting policy, it learns to sit (getup1, getup2) |
| Unity | trajectory gate for single-body rungs, statistical gate for fights (`duel_stats.py` vs `RunDuelStats`) | replay < 1e-4, pelvis < 10 cm; decided share and winning-time quartiles | after first contact two fighters diverge chaotically, so fights are compared as distributions |

## 2. Rewards

- **A win must pay more than surviving to the bell.** r4_a (alive 0.5, upright 0.5, win 10) converged in 130 iterations
  to both fighters leaning on each other for 20 s: a draw paid 40, a win 10. `reward_audit.py` checks this ("stall").
- **A draw that costs the same as a loss invites stalling against a stronger opponent** (cross-body xg2: Trump 65 %
  undecided). Draw -20 (twice a loss) fixed the stall; termination -25 (2.5 x a draw) cut falls without passivity (careful1).
- **The per-step zero clip hides penalties exactly when the policy is doing badly.** It was needed (r0_a: dying early
  beat the penalties; r0_b: the clipped sum was always 0 and only entropy trained), but on the fight rungs it binds on
  18-19 % of steps and erases 12-13 % of the penalty mass (walker and get-up: 0.1 %). Terminal terms and the potential sit
  outside it. Turning it off for a fine-tune did not help: `noclip1` (34 % of steps net negative once the penalties
  reached PPO) lost to careful1 29 / 58 and to the control 38 / 54. Keep the clip; reconsider it only for a ladder from
  scratch.
- **Shaping that is not a potential difference can be farmed.** `ring_advantage` is flat while pushing from the inside
  (r4att_a learned only to stop falling); `push_out` needed a per-episode cap; in the four-way rung `opp_radius` can pay
  32 per round, more than a win (30). The potential version (`r4pbrs`) did not reach careful1 in 400 iterations
  (loses 80 % to it): the hand-shaped terms carried behaviour the potential does not replace quickly.
- **Rewards cannot place limbs.** Three runs tried to get the arms forward with rewards (shover1, shover2, hand_push) and
  failed; building the pose into the action path (style offsets + smoothing) worked. A styled pose switched on at once
  (trump_bull1) or eased in (bull2) made Trump avoid contact (11-13 % wins): a style has to fit how the body already wins.
- **Dominance is normal for some tasks** (get-up: height 73 % of the shaped reward) and a warning for others (Trump's
  fighter: `opp_radius` 64 %).

## 3. Opponents and curricula

- **Frozen opponents must explore like the learner** while training (identical brains: noisy vs clean 35 / 65), unless
  they are rebuilt from ONNX with an arbitrary std, in which case they fall on their own and the learner learns to wait
  (cross-body xg1: 2.4 x more wins in training than losses, the reverse in eval). Use `--frozen-deterministic` then.
- **Warm starts from an exported brain need a critic warm-up** (`--critic-warmup 10-20`): a random value function pushes a
  good actor away from what it does well.
- **Start the pool easy and keep the easy opponents in it.** allround1 started from the brain that beats a standing
  opponent and kept that skill while learning to duel; leagues of chargers alone lost it.
- **The league is not transitive**: beating the current champion is not beating everything before it (allround4 beat
  allround1 74 / 16 but did worse against gens 3-7). Judge a candidate on several opponents (`queue_options.sh suite()`).
- **Clock-driven curricula can fade help before it is used** (getup2). `koth/curriculum.py` advances on the win share
  and steps back below a floor. With the opponent ladder the cross-body runs held their level for the first time
  (Trump vs the G1 league: 31 % wins / 49 % losses against fight1's 33 / 59; Kim 31 / 68 against 33 / 63), where three
  clock-driven recipes ended below their start. Kim needed a step back (stage 1 -> 0 at iteration 210) before she could
  advance. 400 iterations reached stage 2 of 4 for both: the ladder stops the drift, it does not yet add strength.

## 4. Actions

- **The champion's actions sit on the +-1 clamp 29 % of the time while torque is rarely the limit** (1.9 % of ticks):
  the ankles want ~2 rad of roll target against a 0.26 rad range. The humans' brains clamp ~2 %. Audit any new brain
  with `drive_audit.py --ckpt`.
- **Changing the action path of a trained brain is expensive.** 400-iteration fine-tunes of careful1: per-joint scale
  from its own raw outputs (scale1) kept level with the control (vs duel_u1 50 / 45) but lost to careful1 (35 / 63);
  tanh + torque window (mask1) lost 85 % to careful1. These options belong at the start of a new body's ladder, not in a
  champion's fine-tune; the sidecar mechanism (`<name>_action.json`) makes that safe in Unity.
- **One goal law, one set of commands.** Moving the goal point at run time (Tactician) lost to plain Press in every
  variant; training on random tactics (r4tactic) made the brain worse at Press (loses 85 % to careful1) within 400
  iterations. A tactical layer needs its own training budget and its own gate.

## 5. Observations and the Unity mirror

- **Write only what the plugin can mirror** (no keyframes, contact pairs, `ls_iterations` written from C#), use primitive
  colliders, a convex mesh dome rather than an hfield (Warp makes one contact per hfield pair), and expect importer traps:
  ASCII-STL headers, default-class array tails (knee kv read as 0), `mass="0"` ignored on geoms, `OnSyncState` zeroing ctrl.
- **Unity's inputs match training for a body that meets its own kind** (two G1: mean drift <= 0.27 sd, KS <= 0.13).
  **A human meeting other bodies is out of distribution**: Kim in the six-way game, 20 of 112 inputs over the bars
  (opponent distance and command KS 0.43, waist pitch 0.83 sd forward); Trump 8 of 112. Brains trained against their own
  body only: a game of mixed bodies needs cross-body training (section 3).
- **Spawns, noise and the out rule must match** between eval and Unity, or the statistics drift (FFA: more bell ties in
  Unity from spawn spacing and missing observation noise).

## 6. Judging a run

- One variable per run, against the same pool as the run it modifies (`queue_options.sh`: every G1 option repeated
  duel_u1's recipe, so duel_u1 was the control).
- Read the breakdown, not the headline: the "7.3 % self-ejection" was every loss (falls, tripping, the leg-contact rule);
  real self-ejection was 0.2 %.
- `eval.py` flags stall (> 30 % undecided), contact avoidance (< 15 % of time in touch) and self-ejection; a pass with a
  flag still deserves a look. `reward_audit.py` before a long run, `drive_audit.py` and `obs_audit.py` after it.
- Graduation (`graduation.json`) is a floor, not a verdict: export refuses NaN updates and missed reward targets; the
  bars and the Unity gate decide.

## 7. Results of the 2026-10-20 option runs (400 iterations each, 2048 worlds, vs 3 seeds x 256 duels)

| run | change | vs league win / loss | vs careful1 win / loss | vs duel_u1 win / loss | verdict |
|---|---|---|---|---|---|
| duel_u1 (control) | none | 64 / 21 | 39 / 35 | 47 / 40 | |
| scale1 | per-joint action scale | 61 / 32 | 35 / 63 | 50 / 45 | level with the control, below the champion |
| mask1 | tanh map + torque window | 35 / 57 | 10 / 85 | 15 / 85 | rejected |
| pbrs1 | potential instead of ring shaping | 54 / 44 | 19 / 80 | 25 / 72 | rejected |
| tactic1 | random tactics | 29 / 41 | 8 / 85 | 13 / 82 | rejected for Press play |
| trump_xg4 | ladder vs the G1 league | 31 / 49 (fight1 33 / 59) | | | held level; standing G1 59 % |
| kim_xg4 | ladder vs the G1 league | 31 / 68 (fight1 33 / 63) | | | held level; standing G1 81 % |
| noclip1 | no zero clip on the shaped reward | 66 / 28 | 29 / 58 | 38 / 54 | rejected |
