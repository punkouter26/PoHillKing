"""Joint drive calibration audit.

  uv run python scripts/drive_audit.py [--tag koth_2p]                                   # model only
  uv run python scripts/drive_audit.py --ckpt runs/league/careful1.pt --rung r4league    # + action/torque saturation

1. Per joint of robot a_ in model_dump_<tag>.json (the compiled model Unity is gated against): kp, kv, actuator
   force range, ctrl range, damping, armature, next to a reference table: for the G1 the Unitree G1 (29 DoF) joint
   peak torques (as in mujoco_menagerie's g1.xml actuatorfrcrange) and approximate motor speed limits, and the
   leg PD gains of unitree_rl_gym's G1 config; for Kim her own design table (koth/build_kim.py JOINT_CLASSES).
   kp here is the position-actuator gain (N m / rad of ctrl error). The project's leg gains are 2-3 x the
   unitree_rl_gym ones on purpose: lower gains could not hold a static pose in this model (rl_optimization_log.md).
2. With --ckpt: a deterministic rollout on the rung's eval settings (eval.py), per joint: share of policy outputs at
   or beyond the +-1 clamp, mean |action|, share of physics ticks with the actuator force at its limit (>= 99 %),
   and peak |joint velocity| against the speed limit."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT = os.environ.get("KOTH_ROBOT", "g1")
# Unitree G1 29-DoF: peak torque (N m), approximate max speed (rad/s), unitree_rl_gym leg PD (kp, kd); None = not given
G1_SPEC = {
    "hip_pitch": (88, 32, (100, 2)), "hip_roll": (139, 20, (100, 2)), "hip_yaw": (88, 32, (100, 2)), "knee": (139, 20, (150, 4)),
    "ankle_pitch": (50, 37, (40, 2)), "ankle_roll": (50, 37, (40, 2)), "waist_yaw": (88, 32, None), "waist_roll": (50, 37, None),
    "waist_pitch": (50, 37, None), "shoulder_pitch": (25, 37, None), "shoulder_roll": (25, 37, None), "shoulder_yaw": (25, 37, None),
    "elbow": (25, 37, None), "wrist_roll": (25, 37, None), "wrist_pitch": (5, 22, None), "wrist_yaw": (5, 22, None),
}


def kind(joint: str) -> str:
    return joint.replace("left_", "").replace("right_", "").replace("_joint", "")


def spec_table() -> dict:
    if ROBOT == "g1": return G1_SPEC
    from koth.build_kim import JOINT_CLASSES          # class: {name: (axis, range, peak torque, armature)} or one tuple
    out = {}
    for cls, v in JOINT_CLASSES.items():
        for name, t in (v.items() if isinstance(v, dict) else [(cls, v)]):
            out[name.replace("kim_", "")] = (t[2], None, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="koth_2p"); ap.add_argument("--ckpt", default=None); ap.add_argument("--rung", default="r4league")
    ap.add_argument("--opponent", nargs="+", default=None, help="frozen opponents for duel rungs (default: the checkpoint itself)")
    ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dump = json.load(open(os.path.join(ROOT, "assets", ROBOT, f"model_dump_{a.tag}.json")))
    jm = json.load(open(os.path.join(ROOT, "assets", ROBOT, "joint_map.json")))
    joints = {j["name"]: j for j in dump["joints"]}; acts = {x["joint"]: x for x in dump["actuators"]}
    spec = spec_table(); rows = []
    print(f"{ROBOT} model_dump_{a.tag}.json, robot a_ ({'Unitree G1 spec' if ROBOT == 'g1' else 'build_kim design table'} for reference)")
    print(f"{'joint':>28} {'kp':>6} {'kv':>5} {'force':>6} {'spec':>5} {'ctrl range':>15} {'damp':>5} {'arm':>6} {'ref kp/kd':>9}")
    for name in jm["joints"]:
        j = joints["a_" + name]; act = acts["a_" + name]; sp = spec.get(kind(name), (None, None, None))
        frc = act["forcerange"][1]; ok = sp[0] is None or abs(frc - sp[0]) < 1e-6
        ref = f"{sp[2][0]}/{sp[2][1]}" if sp[2] else "-"
        print(f"{name:>28} {act['kp']:6.0f} {act['kv']:5.1f} {frc:6.0f} {sp[0] if sp[0] is not None else '-':>5}{'' if ok else '!'} "
              f"[{act['ctrlrange'][0]:6.2f},{act['ctrlrange'][1]:6.2f}] {j['damping']:5.2f} {j['armature']:6.3f} {ref:>9}")
        rows.append(dict(joint=name, kp=act["kp"], kv=act["kv"], forcerange=act["forcerange"], spec_torque=sp[0], spec_speed=sp[1], ref_pd=sp[2],
                         ctrlrange=act["ctrlrange"], damping=j["damping"], armature=j["armature"], torque_matches_spec=ok))
    if a.ckpt:
        import torch
        from koth.env import KothEnv
        from scripts.eval import eval_cfg, load_policy
        import warp as wp
        opp = a.opponent or ([a.ckpt] if a.rung.startswith(("r4", "r5")) else None)
        env = KothEnv(eval_cfg(a.rung, opponent=opp), a.num_envs, seed=0); policy = load_policy(env, a.ckpt)
        force = wp.to_torch(env.d.actuator_force); aid = env.aid[0]; lim = torch.tensor([r["forcerange"][1] for r in rows], device=env.device)
        spd = torch.tensor([r["spec_speed"] or float("inf") for r in rows], device=env.device)
        clamp = torch.zeros(29, device=env.device); mag = torch.zeros(29, device=env.device); sat = torch.zeros(29, device=env.device)
        vmax = torch.zeros(29, device=env.device); n = 0
        obs = env.reset(); steps = int(a.seconds / env.ctrl_dt)
        for _ in range(steps):
            with torch.no_grad(): act = policy(obs)
            clamp += (act.abs() >= 1).float().sum(0); mag += act.abs().clamp(max=1).sum(0); n += act.shape[0]
            obs, *_ = env.step(act)
            f = force[:, aid].abs(); sat += (f >= 0.99 * lim).float().sum(0) / f.shape[0]
            vmax = torch.maximum(vmax, env._joints()[1][env.lrows if env.frozen is not None else slice(None)].abs().max(0).values)
        clamp, mag, sat = (clamp / n).cpu().numpy(), (mag / n).cpu().numpy(), (sat / steps).cpu().numpy(); vmax = vmax.cpu().numpy()
        print(f"\nrollout: {a.rung}, {a.num_envs} worlds x {steps} steps, deterministic policy {os.path.basename(a.ckpt)}")
        print(f"{'joint':>28} {'|a|>=1':>7} {'mean|a|':>8} {'force at limit':>15} {'peak |qd|':>10} {'speed lim':>10}")
        for i, r in enumerate(rows):
            r.update(action_clamp_share=float(clamp[i]), mean_abs_action=float(mag[i]), force_saturation_share=float(sat[i]), peak_joint_speed=float(vmax[i]))
            print(f"{r['joint']:>28} {clamp[i]:7.1%} {mag[i]:8.2f} {sat[i]:15.1%} {vmax[i]:10.1f} {r['spec_speed'] or '-':>10}")
        print(f"all joints: {clamp.mean():.1%} of actions on the clamp, {sat.mean():.1%} of ticks force-limited")
    if a.json: json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
