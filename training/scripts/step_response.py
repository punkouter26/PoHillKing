"""Joint drive calibration by step response, CPU MuJoCo, for every body (G1, Kim, Trump).

  uv run python scripts/step_response.py [--bodies g1 kim trump] [--step 0.15] [--zeta 0.7] [--json runs/audits/step_response.json]

1. Step test. The pelvis is held in the air (its free joint re-pinned before every physics step, gravity on), every
   joint holds its hold ctrl (joint_map hold_ctrl, the gravity-compensated stance), and one joint's target steps by
   --step rad (toward the middle of its range). 0.5 s at the training timestep. Per joint: rise time 10-90 %, overshoot,
   settling time (within 5 % of the step), steady-state error (last 0.1 s), and whether the step met the torque limit.
   Model numbers: the joint's effective inertia I = M_jj at the hold pose (mj_fullM diagonal, armature included),
   natural frequency wn = sqrt(kp / I), damping ratio zeta = kv / (2 sqrt(kp I)).
2. Symmetry. Standing on the floor with the hold ctrl for 2 s: the actuator torque each joint needs, left against right.
   A body whose mirrored joints carry different loads has an asymmetric mass model (or stance).
3. Suggested kv for --zeta at the same kp, where the measured zeta is outside [0.3, 1.5]; reported, never applied: a gain
   change changes the physics every brain of that body was trained in, so it goes with a retrain."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from koth import cpu_sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = mujoco.mjtObj.mjOBJ_JOINT


def load(body):
    spec = cpu_sim.joint_map(body)
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets", body, "scene_flat_1p_train.xml")); d = mujoco.MjData(m)
    s = cpu_sim.robot(m, spec["joints"], "a_"); j = [mujoco.mj_name2id(m, J, "a_" + n) for n in spec["joints"]]
    return spec, m, d, s["qadr"], s["dadr"], s["aid"], s["rq"], s["rd"], j


def hold_state(spec, m, d, qa, aid, rq, z):
    mujoco.mj_resetDataKeyframe(m, d, 0); d.qpos[rq + 2] = z
    d.qpos[qa] = spec["default_pose"]; d.ctrl[aid] = spec["hold_ctrl"]; mujoco.mj_forward(m, d)


def step_test(body, step, t_end=0.5):
    spec, m, d, qa, da, aid, rq, rd, jid = load(body); out = []
    hold_state(spec, m, d, qa, aid, rq, spec["key_root_z"] + 0.5)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M); inertia = M[da, da]
    root0 = d.qpos[rq:rq + 7].copy()
    for i, name in enumerate(spec["joints"]):
        hold_state(spec, m, d, qa, aid, rq, spec["key_root_z"] + 0.5)
        lo, hi = m.jnt_range[jid[i]]; q0 = d.qpos[qa[i]]; s = step if (hi - q0) >= (q0 - lo) else -step
        s = float(np.clip(q0 + s, lo + 0.02, hi - 0.02) - q0)
        ctrl = np.array(spec["hold_ctrl"]); target = ctrl[i] + s; ctrl[i] = target
        kp = m.actuator_gainprm[aid[i], 0]; kv = -m.actuator_biasprm[aid[i], 2]; tau = m.actuator_forcerange[aid[i], 1]
        n = int(t_end / m.opt.timestep); qs = np.zeros(n); sat = 0
        for k in range(n):
            d.qpos[rq:rq + 7] = root0; d.qvel[rd:rd + 6] = 0; d.ctrl[aid] = ctrl
            mujoco.mj_step(m, d); qs[k] = d.qpos[qa[i]]
            sat += abs(d.actuator_force[aid[i]]) >= 0.99 * tau
        y = (qs - q0) / s; t = np.arange(1, n + 1) * m.opt.timestep
        rise = (t[np.argmax(y >= 0.9)] - t[np.argmax(y >= 0.1)]) if (y >= 0.9).any() else float("nan")
        outside = np.nonzero(np.abs(y - 1) > 0.05)[0]; settle = t[outside[-1]] if len(outside) and outside[-1] < n - 1 else (0.0 if not len(outside) else float("nan"))
        I = float(inertia[i]); wn = float(np.sqrt(kp / I)); zeta = float(kv / (2 * np.sqrt(kp * I)))
        out.append(dict(joint=name, step=s, kp=float(kp), kv=float(kv), tau=float(tau), inertia=I, wn=wn, zeta=zeta,
                        rise_s=float(rise), overshoot=float(max(y.max() - 1, 0)), settle_s=float(settle), ss_error=float(abs(1 - y[-50:].mean())),
                        torque_limited_share=sat / n))
    return out, spec


def symmetry(body, seconds=2.0):
    spec, m, d, qa, da, aid, rq, rd, _ = load(body)
    hold_state(spec, m, d, qa, aid, rq, spec["key_root_z"]); tau = np.zeros(len(aid)); n = int(seconds / m.opt.timestep)
    for k in range(n):
        d.ctrl[aid] = spec["hold_ctrl"]; mujoco.mj_step(m, d)
        if k >= n // 2: tau += d.actuator_force[aid]
    tau /= n - n // 2; up = cpu_sim.upright(d.qpos[rq + 3:rq + 7])
    names = spec["joints"]; rows = []
    for i, nm in enumerate(names):
        if not nm.startswith("left_"): continue
        r = names.index(nm.replace("left_", "right_")); sign = -1.0 if any(k in nm for k in ("roll", "yaw")) else 1.0   # roll/yaw mirror with a sign flip
        rows.append(dict(joint=nm.replace("left_", ""), left=float(tau[i]), right=float(tau[r]), mismatch=float(abs(tau[i] - sign * tau[r]))))
    com = d.subtree_com[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_pelvis")]
    feet = [d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"a_{s}_ankle_roll_link")] for s in ("left", "right")]
    return dict(rows=rows, up=float(up), com_lateral_offset=float(com[1] - (feet[0][1] + feet[1][1]) / 2), pelvis_drift=float(np.hypot(*d.qpos[rq:rq + 2])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bodies", nargs="+", default=["g1", "kim", "trump"])
    ap.add_argument("--step", type=float, default=0.15); ap.add_argument("--zeta", type=float, default=0.7)
    ap.add_argument("--json", default=os.path.join(ROOT, "runs", "audits", "step_response.json"))
    a = ap.parse_args(); res = {}
    for body in a.bodies:
        rows, spec = step_test(body, a.step); sym = symmetry(body)
        for r in rows:
            r["suggest_kv"] = round(2 * a.zeta * np.sqrt(r["kp"] * r["inertia"]), 2) if not 0.3 <= r["zeta"] <= 1.5 else None
        res[body] = dict(step=rows, symmetry=sym)
        print(f"\n== {body}: step {a.step} rad, pelvis held")
        print(f"{'joint':>28} {'kp':>6} {'kv':>5} {'I':>7} {'wn':>6} {'zeta':>5} {'rise':>6} {'over':>6} {'settle':>6} {'sserr':>6} {'tau-lim':>7} {'kv@' + str(a.zeta):>7}")
        for r in rows:
            print(f"{r['joint']:>28} {r['kp']:6.0f} {r['kv']:5.1f} {r['inertia']:7.4f} {r['wn']:6.1f} {r['zeta']:5.2f} {r['rise_s'] * 1e3:5.0f}ms {r['overshoot']:6.1%} "
                  f"{r['settle_s'] * 1e3:5.0f}ms {r['ss_error']:6.1%} {r['torque_limited_share']:7.1%} {r['suggest_kv'] if r['suggest_kv'] is not None else '-':>7}")
        print(f"standing hold, 2 s: up {sym['up']:.4f}, centre of mass {sym['com_lateral_offset'] * 1e3:+.1f} mm lateral of the feet midpoint, pelvis drift {sym['pelvis_drift'] * 1e3:.1f} mm")
        print(f"{'joint':>20} {'left Nm':>8} {'right Nm':>8} {'mismatch':>8}")
        for r in sym["rows"]: print(f"{r['joint']:>20} {r['left']:8.2f} {r['right']:8.2f} {r['mismatch']:8.2f}")
    os.makedirs(os.path.dirname(a.json), exist_ok=True); json.dump(res, open(a.json, "w"), indent=1); print("\nwrote", a.json)


if __name__ == "__main__":
    main()
