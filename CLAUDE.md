# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## What this is

PoKingHill: king-of-the-hill between physics-simulated humanoids (Unitree G1, plus two human bodies: Kim, a scanned 56 kg woman, and Trump, a 102 kg man from a rigged GLB). Brains are trained in Python on MuJoCo Warp and run in Unity 6000.6.0f1 through the `org.mujoco` plugin and `com.unity.ai.inference` (Sentis' successor). **PhysX is never used**: Unity's physics mode is Script, every scene must have zero Collider/Rigidbody, and MuJoCo is the only simulator in both worlds.

`tasks.md` is the build contract (phases 0, A-D, rung bars R0-R5). `rl_optimization_log.md` is the chronological record of every run, measurement and decision, newest at the bottom. Read the bottom of it before changing training or parity tooling.

## Commands

Python (from `training/`, uv project, Python 3.11, CUDA torch). The `.sh` scripts call `.venv/Scripts/python.exe` directly.

```bash
uv run python scripts/train.py --rung r0 --num-envs 4096 --iters 1500 [--resume ckpt.pt | --warm ckpt.pt] [--opponent a.pt b.pt:stand] [--reward self_edge=-5]
uv run python scripts/eval.py --rung r4league --ckpt runs/league/careful1.pt --opponent runs/league/gen3.pt --seeds 3 --num-envs 256
bash scripts/judge.sh <league checkpoint name>       # champion bars: vs standing walker, gens 3-7, champion, itself
uv run python scripts/export_policy.py --ckpt <ckpt.pt> --name attacker   # -> Assets/PoKingHill/Models/<name>_policy.onnx
uv run python scripts/duel_stats.py --policy attacker --rounds 200 [--getup]   # CPU MuJoCo reference for the Unity statistics gate (--getup: the get-up rule)
uv run python scripts/getup_stats.py --policy getup --drops 200          # CPU reference for the get-up gate (RunGetupStats)
uv run python scripts/export_mlp.py                  # roster brains' weights as <name>_mlp.bytes for the Tactician's rollouts (re-run after exporting a brain)
uv run python scripts/collision_audit.py             # self-collision of G1/Kim/Trump in zero, rest, T-pose and swing arcs -> SCREENSHOTS/collision-<YYMMDD>/report.html
uv run python -m koth.build_mjcf [4p|5p]             # regenerate MJCF scenes, model_dump_*.json and joint_map.json
uv run python -m koth.env                            # env smoke test
```

`KOTH_ROBOT=kim` or `KOTH_ROBOT=trump` switches every Python entry point (and `export_policy.py`'s output folder to `Models/<robot>/`) to that human's body; G1 is the default. Human rungs are `<robot>_r0b` / `<robot>_r1b` (walking recipe, `env.human_overrides`) and `trump_r4bull` (Trump's fight). Run one 4096-env training at a time: two plus an eval fill the 12 GB GPU and slow everything ~100x. TensorBoard logs and checkpoints go to `training/runs/` (git-ignored; `runs/league/champion.txt` names the current champion).

Unity: batch mode does **not** work (no headless licence). Launch a normal editor with `-executeMethod` and `-kothExit`; the editor must be signed in via `unity auth login` / `unity license activate`, otherwise it exits with code 198.

```bash
"/c/Program Files/Unity/Hub/Editor/6000.6.0f1/Editor/Unity.exe" -projectPath . -executeMethod PoKingHill.EditorTools.ParityBatch.<Method> [args] -kothExit
```

`ParityBatch` methods ([Assets/PoKingHill/Editor/ParityBatch.cs](Assets/PoKingHill/Editor/ParityBatch.cs)):
- `ImportAll` / `ImportOne -kothScene koth_6p`: import `training/assets/<robot>/scene_<tag>_unity.xml` into `Testbed_<tag>.unity` (`-kothRobot kim|trump` for a human body).
- `RunHold -kothScene flat_1p`, `RunPolicy -kothScene <tag> -kothRung r1`: trajectory parity gates against `<rung>_reference_trajectory*.json`.
- `RunDuelStats -kothRounds 100 -kothA 0 -kothB 0`, `RunDuelParity`: combat gates (statistical, on koth_2p). `RunGetupDuelStats` (the get-up rule, vs `duel_stats.py --getup`), `RunGetupStats` (drops on flat_1p, vs `getup_stats.py`), `RunTacticianStats` (A/B: seat A plans, B does not; overload with horizon / Press prior / candidate count for tuning from `eval`).
- `BuildDuelDemo`, `PlayDuelDemo` (add `-kothShots <dir>` for menu/combat/ejection captures), `BuildPlayer` (`Builds/Win`, `-kothDev`), `MapOverview`.
- Player switches: `-kothRounds N` (stats gate in the player), `-kothPerf 30` (writes `perf_unity.json`).

When an editor already has the project open, drive it with the Unity CLI instead of launching a second one (it needs `com.unity.pipeline` ≥ 0.6; the editor must leave play mode to resolve packages or recompile):

```bash
U="$LOCALAPPDATA/Unity/bin/unity.exe"; P="C:/Users/punko/Downloads/PoHillKing"
"$U" command recompile --project-path $P                     # then poll: recompile_status
"$U" command eval --code 'PoKingHill.EditorTools.ParityBatch.RunDuelStats(); return "ok";' --project-path $P
"$U" command run_tests --mode editor --filter PoKingHill.Tests.Editor --filter_type assembly --project-path $P
"$U" command run_tests --mode playmode --filter PoKingHill.Tests.PlayMode --filter_type assembly --async_tests true --project-path $P   # poll: test_status
```
`eval` bodies run inside a method (no `using`; return a string). Compile errors appear only in `Logs/Editor.log`. When the Device Simulator window is open it is the main play view and `Screen` follows the simulated phone; captures with `ScreenCapture` come from the Game view.

Tests (`Assets/PoKingHill/Tests`): EditMode `PhysXAudit` (no PhysX components, simulation mode Script, fixed timestep 0.002) and `GameSceneTests` (game scene has no missing scripts and every reference wired, arena rule = training constants, model dumps list their robots, Elo, build version); PlayMode `MatchFlowTests` (Menu → Fight → Result → Menu in the real scene). `ModelDumpCheck` runs on scene init and diffs the compiled mjModel against `model_dump_<tag>.json`.

## Architecture

**One model, two simulators.** `koth/build_mjcf.py` is the single source of the physics model: it writes the training MJCF, a `_unity.xml` variant stripped of what the plugin cannot import, `model_dump_<tag>.json` (the parity reference) and `joint_map.json` (canonical 29-joint order). Unity scenes are *imported* from that XML by `ParityBatch`, never hand-built, and `Demo_duel.unity` is regenerated by `BuildDuelDemo`. Change physics on the Python side and re-import; never edit imported MuJoCo components in Unity. Settings the plugin cannot import (`ls_iterations`, keyframes, contact pairs) are either avoided in the MJCF or written from C# after init.

**Control loop, mirrored on both sides.** 2 ms physics step, policy every 10th step (50 Hz). Observations: 103 locomotion inputs (`koth/obs.py:build_obs` = `ObsBuilder.cs`), plus 9 combat inputs for 112-dim brains (`build_combat` = `ObsBuilder.FillCombat`). The 3 command slots come from a geometric goal law (`obs.goal_command` = `GoalCommand.cs`), not from extra observations. `ctrl = default + 0.5·action`. Any change to one side of these pairs must be made on the other, then re-gated.

**Unity runtime** ([Assets/PoKingHill/Scripts](Assets/PoKingHill/Scripts)): everything hooks `MjScene` events (`postInitEvent`, `preUpdateEvent`, `postUpdateEvent`) and reads/writes `mjData` through unsafe pointers. No Instantiate/Destroy, no scene reloads: rounds reset by writing qpos/qvel. `PolicyRunner` must rewrite held ctrl in `preUpdateEvent` every step because `MjActuator.OnSyncState` zeroes ctrl after each `mj_step`. The physics tick is allocation-free; keep it that way.
- `DemoDirector`: the match state machine (`Menu → Fight → Result (→ Replay) → Fight | Menu`); quick match, custom, tournament, puppet seat. Seats agents on bodies of their own kind (`g1`/`kim`/`trump`), reorders `fighters[]` at every launch (cache your own copy if you index by body), parks unused bodies 50 m up, points each brain at one opponent (nearest, or with `Targeting.Opportunist` the one easiest to push off; brains were trained 1v1). The statistics gate (`statsRounds`) bypasses Result and the presentation.
- Presentation never gets called by the director: it subscribes to `MatchEvents` (state, round start/end, knock-out and its credit, down/recovered, hype, impact, footfall, splash, caption, commentary). `HudView` (UI Toolkit, styles in `UI/Hud.uss`, one 9:16 column, no scrolling; badges carry a `VitalsBar`, knock-out feed, intro card, tabs Play/Agents/Match/Show/Connect), `BroadcastCamera` (Cinemachine; intro crane on a Splines path + per-fighter orbit) or `MatchCamera` (Classic), `ReplaySystem` (disables `MjScene`, replays qpos via `mj_kinematics`, restores the exact state), `MatchFx` (particles, decals, post punch, shockwave via the `ShockwaveFeature` renderer feature, lightning, danger light), `BodyLook` (team paint, overlays, and the `FX/Fighter.shader` surface: seat-colour rim that turns red at the edge, wear from contacts, wet after the sea, hit flash), `TimeWarp` (slow motion at the rim and hit stop: `Time.timeScale` only, never in gates or replays), `QualityTier` (Low/Medium/High/Auto presets on the camera and sun, frame-time telemetry), `WinOdds`, `Commentator` (+ `VoiceCaster`, Azure Speech / ElevenLabs), `Hazards` + `ChatVotes` (+ `!hype`), `WeatherFeed` (Open-Meteo), `MatchLog` (telemetry JSONL + Elo `Ladder` with record, KOs, assists in `persistentDataPath`). Options live in `GameSettings` (PlayerPrefs, plain defaults).
- Match state `Intro` (tale of the tape) sits between Menu and Fight on a match's first round: bodies stand passively (no brain), then `ArmBrains` + `NewRound` place them afresh. Gates and auto runs skip it; tests call `SkipIntro()`.
- Sensors and mechanics, all inert in the statistics gates unless a gate asks: `ContactLedger` (pairwise pushing impulse from `mj_contactForce`, knock-out credit and assists, Grudge/Vulture targeting), `Biometrics` (capture point vs the planted feet = balance, capture point vs the rim = edge, strain history), `Stamina` (kp/bias scale 0.85-1.15 from joint power, inside the trained ±20 % kp randomisation; restores nominal gains when off), `Tactician` (rollout planner: copies `mjData`, simulates candidate tactics on worker threads with `MlpNet` = the ONNX brains' weights from `export_mlp.py`; tactics only move the goal point, `Tactics.Goal`). The get-up rule lives in `DemoDirector.OutWithGetUp` (body-kind brains in `getupBrains`, `PolicyRunner.SetGetUp`), mirrored by `duel_stats.py --getup`.
- `Sea` (rising water = round clock, buoyancy via `xfrc_applied` for every fighter body), `ImpactSynth` (procedural audio, no audio files: hits by material and body through `SoundPool`'s 3D one-shots whose clips are synthesized at start, sliding and grinding from the contact list, ambience beds of wind, surf, crowd and quake rumble in `OnAudioFilterRead`; mix channels in `GameSettings`), `SkinFollower` (a human's mesh follows its MuJoCo bodies), `ProjectilePool`.
- Parity/diagnostic components (`ParityProbe`, `PolicyParityProbe`, `ModelDumpCheck`, `PerfProbe`, `AutoShot`) are inert unless their `-koth*` switch is given; switches are read through `Cli` (the editor's own command line: a CLI `eval` cannot set them, so gates that need parameters have overloads). `GetupProbe` is added only by `RunGetupStats`.

**Scenes.** `Testbed_<tag>.unity` per imported model (`flat_1p`, `koth_1p/2p/4p/5p/6p`, `kim_*`, `trump_*`); the robots come from the `robots` list in `model_dump_<tag>.json`. The game is `Demo_duel.unity`, built from `koth_6p` (four G1, Kim as `e_`, Trump as `f_`; `python -m koth.build_mjcf 6p`) by `BuildDuelDemo`, which calls `GameSceneBuilder.Dress` for the HUD, cameras, effects and their shared assets (`UI/`, `FX/`). The two-fighter gates build `Gate_duel_2p.unity` (git-ignored) from `koth_2p`. The HUD's version is YYMMDD of the last commit (`Editor/BuildVersion.cs` → `Resources/build_version.txt`, git-ignored).

**Gates.** Training checkpoints are only trusted after the Unity gate: trajectory parity for single-agent rungs (replay error < 1e-4, closed-loop pelvis < 10 cm), statistical parity for combat (decided share and winning-time quartiles vs `duel_stats.py`), because adversarial rounds diverge chaotically after contact. Known rules mismatch: training/`eval.py` end a round on own foot-shin contact, Unity and `duel_stats.py` do not (`eval.py --leg-contact-rule unity` measures the Unity rule; staged rungs turn it off). Staged next steps and their Unity mirrors: `training/STAGED_TRAINING.md`.

**Art.** Blender scripts generate render-only assets: `tools/make_mountain_map.py` → `Assets/PoKingHill/Maps/Mountain` (Poly Haven CC0, git-ignored sources in `tools/polyhaven`; hill in three LODs, rocks, crags, ruins, banners swaying in `FX/Banner.shader` from MatchFx's `_KothWind`, Cycles-baked AO; headless `blender -b --factory-startup --python tools/make_mountain_map.py`, the Store build via `%LOCALAPPDATA%/Microsoft/WindowsApps/blender-launcher.exe`, which does not pass stdout back; archives `Assets/Art/Raw/Mountain~/Mountain.blend` and `Assets/Art/Exports/GLB/Mountain.glb`), `tools/make_kim.py` (reads `Assets/Art/Raw/Kim/Kim.glb`) → `Assets/PoKingHill/Fighters/Kim` + `training/assets/kim/kim_rig.json`, then `koth/build_kim.py` (her profile) and `koth/build_human.py` (the builder shared by every human) build her MJCF with the G1's joint names and order so networks and runners are shared. A rigged GLB needs no Blender: `tools/make_glb_fighter.py <robot>` reads `Assets/Art/Raw/<Name>/<Name>.glb`, writes `<robot>_rig.json` and a 2k-texture copy in `Fighters/<Name>/` (imported by glTFast), and `koth/build_<robot>.py` holds the body's mass, strength and gains (Trump: `koth/build_trump.py`).

**Line endings.** The index mixes CRLF and LF files (`core.autocrlf=true`). `sed -i` and some tools rewrite CRLF files as LF, which turns a small edit into a whole-file diff; check `git ls-files --eol -m` and restore CRLF (`sed -i 's/\r*$/\r/'`) before committing.
