"""Outcome statistics for attacker-vs-attacker rounds in CPU MuJoCo, with the same rules and spawn recipe as Unity's
MatchDirector (both on the summit at radius 0.5-1.2 m on opposite bearings, random yaw; out = koth.rules.is_out:
radius > 1.7 m, fallen, or the 25 s bell). Used for the statistical Unity parity gate of the combat rungs.

  uv run python scripts/duel_stats.py --policy attacker --rounds 200 [--getup]
Writes training/assets/<robot>/duel_stats_python.json (policies from Assets/PoKingHill/Models[/<robot>]).

--getup plays the get-up rule (MatchDirector.OutWithGetUp, ParityBatch.RunGetupDuelStats): a ring-out is out at
once; a fall inside the plateau (radius < 1.5 m) hands the robot to the get-up brain (getup_policy.onnx, zero command)
and starts a 5 s count; standing (up > 0.9, pelvis above 85 % of the standing height) for 0.4 s gives it back to the
fight brain, the count running out is a knock-out; at most 2 get-ups per robot per round. Writes
duel_stats_getup_python.json.
--getup --hill-rule plays the game's rule since 2026-10-21 (RunGetupDuelStats with hill): every fall anywhere inside
the out radius gets up, no count, no limit; only a ring-out is out. Writes duel_stats_hill_python.json.

A brain exported with <name>_action.json (koth/action_map.py: tanh map, per-joint scale, torque window) is run through
that action path, as Unity's PolicyRunner does."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from koth.obs import goal_command, STYLE_POSES, SHOVE_FAR, SHOVE_NEAR, SHOVE_SMOOTH
from koth.build_mjcf import ASSETS, UNITY_MODELS
from koth.rules import is_out, ground_z, OUT_RADIUS, PLATEAU_R, FALL_UP, FALL_HEIGHT
from koth import cpu_sim
from koth.action_map import NumpyMap, read_sidecar

ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="attacker"); ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--seconds", type=float, default=25.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shove", action="store_true", help="both robots use the shove style (arm pose + action smoothing), as in koth/env.py")
ap.add_argument("--style-pose", default="shove", choices=list(STYLE_POSES), help="the styled rest pose with --shove (bull: Trump's charge)")
ap.add_argument("--vmax", type=float, default=1.0, help="goal-law walking speed, as the roster entry (Trump: 0.7)")
ap.add_argument("--getup", action="store_true", help="the game's get-up rule with getup_policy.onnx (see the docstring)")
ap.add_argument("--getup-seconds", type=float, default=5.0); ap.add_argument("--max-getups", type=int, default=2)
ap.add_argument("--hill-rule", action="store_true", help="with --getup: the game's rule (no count, no limit, anywhere inside the out radius)")
a = ap.parse_args()
if a.hill_rule: a.getup = True; a.getup_seconds = 0.0; a.max_getups = 0
spec = cpu_sim.joint_map()
policy_path = os.path.join(UNITY_MODELS, f"{a.policy}_policy.onnx")
sess = cpu_sim.session(policy_path)
m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "scene_koth_2p_train.xml")); d = mujoco.MjData(m)
R = cpu_sim.robots(m, spec["joints"], ("a_", "b_"))
for s_ in R: s_["amap"] = NumpyMap(read_sidecar(policy_path), spec["action_scale"], *NumpyMap.gains(m, s_["aid"]))
pool = cpu_sim.pool(m, spec["pool_bodies"])
default = np.array(spec["default_pose"])
shove_offset = np.array([next((v - default[i] for k, v in STYLE_POSES[a.style_pose].items() if k in j), 0.0) for i, j in enumerate(spec["joints"])])
rng = np.random.default_rng(a.seed)
T = cpu_sim.T
getup_sess = cpu_sim.session(os.path.join(UNITY_MODELS, "getup_policy.onnx")) if a.getup else None
key_z = spec["key_root_z"]


def out_with_getup(s, t):
    """MatchDirector.OutWithGetUp for robot s at round time t (s["down"], s["stood"], s["getups"] carry the state)."""
    pos, quat = d.qpos[s["rq"]:s["rq"] + 3], d.qpos[s["rq"] + 3:s["rq"] + 7]
    rad = np.hypot(pos[0], pos[1]); h = pos[2] - ground_z(rad); up = cpu_sim.upright(quat)
    if rad > OUT_RADIUS: return True
    if s["down"] is not None:
        if up > 0.9 and h > 0.85 * key_z:
            s["stood"] = t if s["stood"] is None else s["stood"]
            if t - s["stood"] >= 0.4: s["down"] = s["stood"] = None
        else: s["stood"] = None
        return s["down"] is not None and a.getup_seconds > 0 and t - s["down"] > a.getup_seconds
    if not (up < FALL_UP or h < FALL_HEIGHT): return False
    if (a.hill_rule or rad < PLATEAU_R) and (a.max_getups <= 0 or s["getups"] < a.max_getups):
        s["down"] = t; s["stood"] = None; s["getups"] += 1; return False
    return True
wins = [0, 0]; ties = 0; times = []
for rnd in range(a.rounds):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    bearing = rng.uniform(0, 2 * np.pi)
    for i, s in enumerate(R):
        r = rng.uniform(0.5, 1.2); b = bearing + i * np.pi; yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[s["rq"]:s["rq"] + 2] = [r * np.cos(b), r * np.sin(b)]; d.qpos[s["rq"] + 3:s["rq"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        s["last"] = np.zeros(29, dtype=np.float32); s["f"] = np.zeros(29); s["phase"] = np.float32(0)
        s["down"] = None; s["stood"] = None; s["getups"] = 0
    park = cpu_sim.park(d, pool)
    result = None
    for tick in range(int(a.seconds / spec["ctrl_dt"])):
        for i, s in enumerate(R):
            o = R[1 - i]; rq = s["rq"]; quat = d.qpos[rq + 3:rq + 7]
            if s["down"] is not None:              # the get-up brain drives (PolicyRunner.GetUpTick): zero command, no style path
                obs = cpu_sim.observe(d, s, default, np.zeros(3), s["phase"])[0].numpy().astype(np.float32)
                act = np.clip(cpu_sim.infer(getup_sess, obs), -1, 1).astype(np.float32)
                s["ctrl"] = cpu_sim.default_ctrl(s, default, spec["action_scale"], act); s["last"] = act; s["f"] = act.astype(np.float64)
                s["phase"] = cpu_sim.advance_phase(s["phase"], spec["ctrl_dt"])
                continue
            cmd = goal_command(T(quat), T([d.qpos[o["rq"]] - d.qpos[rq], d.qpos[o["rq"] + 1] - d.qpos[rq + 1]]), 0.0, a.vmax)[0].numpy()
            obs = cpu_sim.observe(d, s, default, cmd, s["phase"], other=o)[0].numpy().astype(np.float32)
            act = s["amap"].to_action(cpu_sim.infer(sess, obs))
            f = act.astype(np.float64); off = 0.0
            k = max(SHOVE_SMOOTH if a.shove else 0.0, s["amap"].smooth)       # action low-pass (action_map.py smooth)
            if k > 0: f = s["f"] = k * s["f"] + (1 - k) * f
            if a.shove:
                dist = np.hypot(d.qpos[o["rq"]] - d.qpos[rq], d.qpos[o["rq"] + 1] - d.qpos[rq + 1])
                off = np.clip((SHOVE_FAR - dist) / (SHOVE_FAR - SHOVE_NEAR), 0, 1) * shove_offset
            s["ctrl"] = s["amap"].target(default + off, f, d.qpos[s["qadr"]], d.qvel[s["dadr"]], s["lo"], s["hi"]); s["last"] = act
            s["phase"] = cpu_sim.advance_phase(s["phase"], spec["ctrl_dt"])
        cpu_sim.step(m, d, pool, park, R, spec["decimation"])
        t_now = (tick + 1) * spec["ctrl_dt"]
        out = [out_with_getup(s, t_now) if a.getup else is_out(d.qpos[s["rq"]:s["rq"] + 3], d.qpos[s["rq"] + 3:s["rq"] + 7]) for s in R]
        if any(out):
            if out[0] != out[1]: result = 1 if out[0] else 0; wins[result] += 1; times.append((tick + 1) * spec["ctrl_dt"])
            else: ties += 1
            break
    else:
        ties += 1
    if (rnd + 1) % 25 == 0: print(f"{rnd + 1} rounds: A {wins[0]}  B {wins[1]}  ties {ties}", flush=True)
res = dict(rounds=a.rounds, wins_a=wins[0], wins_b=wins[1], ties=ties, decided=round((wins[0] + wins[1]) / a.rounds, 4),
           median_time_s=round(float(np.median(times)), 2) if times else -1, mean_time_s=round(float(np.mean(times)), 2) if times else -1,
           p25_time_s=round(float(np.percentile(times, 25)), 2) if times else -1, p75_time_s=round(float(np.percentile(times, 75)), 2) if times else -1)
res["getup_rule"] = "hill" if a.hill_rule else a.getup
out_name = "duel_stats_hill_python.json" if a.hill_rule else "duel_stats_getup_python.json" if a.getup else "duel_stats_python.json"
json.dump(res, open(os.path.join(ASSETS, out_name), "w"), indent=1); print(json.dumps(res))
