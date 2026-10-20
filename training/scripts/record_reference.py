"""Export a checkpoint to a single-file ONNX (opset 17, batch 1) and record the parity reference trajectory.

  uv run python scripts/record_reference.py --rung r1 --ckpt runs/r1/r1_b/model_final.pt --cmd 0.6 0 0.4
  uv run python scripts/record_reference.py --rung r2 --ckpt ... --scene scene_koth_1p_train.xml --goal center --spawn 2.4 0.7 1.0
  uv run python scripts/record_reference.py --rung r3 --ckpt ... --scene scene_koth_2p_train.xml --robots a_ b_ --goal opponent

The reference is produced by CPU MuJoCo (double precision, the same engine the Unity plugin runs) driven by the
exported ONNX through onnxruntime, with the exact tick order of Unity's PolicyRunner: build obs, infer, write
ctrl, then 10 physics steps. No noise, no domain randomization, no pushes; pool boxes pinned every physics step.
Outputs into Assets/PoKingHill/Models/:  <rung>_policy.onnx   <rung>_reference_trajectory[_<robot>].json
Checks: torch policy vs ONNX max abs error < 1e-5 on 1000 random observations."""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, mujoco
from koth.obs import goal_command, OBS_DIM
from koth.build_mjcf import ASSETS, UNITY_MODELS    # Models/ for G1, Models/<robot>/ otherwise
from koth.rules import PLATEAU_R, SLOPE_K
from koth import cpu_sim


def export_onnx(ckpt: str, out_path: str) -> None:
    """Checkpoint actor -> single-file ONNX at out_path (scripts/export_policy.py, which checks torch vs ONNX)."""
    from scripts.export_policy import export
    export(ckpt, out_path)


def record(onnx_path, scene, prefixes, cmd, seconds, goal="none", spawns=None, stop_dist=0.3, vmax=0.8) -> dict:
    # onnx_path, goal, stop_dist and vmax may each be one value or one per robot (a duel has two different brains)
    per = lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * len(prefixes)
    onnx_paths, goals, stops, vmaxs = per(onnx_path), per(goal), per(stop_dist), per(vmax)
    """Returns {prefix: reference dict}. spawns: {prefix: (r, bearing, yaw)} or None for the keyframe pose."""
    spec = cpu_sim.joint_map()
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, scene)); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    R = {}
    for p in prefixes:
        R[p] = dict(cpu_sim.robot(m, spec["joints"], p), last=np.zeros(29, dtype=np.float32), phase=np.float32(0.0), frames=[],
                    cmd=np.array(cmd, dtype=np.float32), ctrl=None)
        if spawns and p in spawns:               # same placement rule as KothEnv.reset_idx
            r, bearing, yaw = spawns[p]; rq = R[p]["rq"]
            slope = 2 * SLOPE_K * max(r - PLATEAU_R, 0.0); h = -SLOPE_K * max(r - PLATEAU_R, 0.0) ** 2
            d.qpos[rq:rq + 3] = [r * np.cos(bearing), r * np.sin(bearing), spec["key_root_z"] + h + 0.02 + 0.12 * slope]
            d.qpos[rq + 3:rq + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    pool = cpu_sim.pool(m, spec["pool_bodies"])
    park = cpu_sim.park(d, pool)
    default = np.array(spec["default_pose"], dtype=np.float32)       # float32-rounded default pose, as Unity holds it
    sessions = [cpu_sim.session(p_) for p_ in onnx_paths]
    f32 = lambda a: [float(np.float32(v)) for v in a]; f64 = lambda a: [float(v) for v in a]
    T = cpu_sim.T
    for tick in range(int(round(seconds / spec["ctrl_dt"]))):
        for pi, p in enumerate(prefixes):         # all robots observe the same pre-step state, as in Unity
            s = R[p]; rq, rd = s["rq"], s["rd"]; goal = goals[pi]; sess = sessions[pi]
            other = R[[q for q in prefixes if q != p][0]] if len(prefixes) == 2 else None
            if goal == "center":
                gv = [-d.qpos[rq], -d.qpos[rq + 1]]
            elif goal == "opponent":
                o = other["rq"]; gv = [d.qpos[o] - d.qpos[rq], d.qpos[o + 1] - d.qpos[rq + 1]]
            if goal != "none":
                s["cmd"] = goal_command(T(d.qpos[rq + 3:rq + 7]), T(gv), stops[pi], vmaxs[pi])[0].numpy().astype(np.float32)
            combat = sess.get_inputs()[0].shape[1] > OBS_DIM          # combat policy: append the opponent/ring block
            obs = cpu_sim.observe(d, s, default, s["cmd"], s["phase"], other=other if combat else None)[0].numpy().astype(np.float32)
            raw = cpu_sim.infer(sess, obs)
            act = np.clip(raw, -1, 1).astype(np.float32)
            s["ctrl"] = cpu_sim.default_ctrl(s, default, spec["action_scale"], act)
            s["frames"].append(dict(t=float(d.time), root_pos=f64(d.qpos[rq:rq + 3]), root_quat=f64(d.qpos[rq + 3:rq + 7]),
                                    root_linvel=f64(d.qvel[rd:rd + 3]), root_angvel=f64(d.qvel[rd + 3:rd + 6]),
                                    joint_pos=f64(d.qpos[s["qadr"]]), joint_vel=f64(d.qvel[s["dadr"]]),
                                    obs=f32(obs), action=f32(raw), ctrl=f64(s["ctrl"]), command=f32(s["cmd"])))
            s["last"] = act; s["phase"] = cpu_sim.advance_phase(s["phase"], spec["ctrl_dt"])
        cpu_sim.step(m, d, pool, park, [R[p] for p in prefixes], spec["decimation"])
    out = {}
    for p in prefixes:
        fr = R[p]["frames"]; up = [cpu_sim.upright(f["root_quat"]) for f in fr]
        rad = [float(np.hypot(*f["root_pos"][:2])) for f in fr]
        print(f"{p} reference: {len(fr)} ticks, pelvis z {fr[0]['root_pos'][2]:.4f} -> {fr[-1]['root_pos'][2]:.4f}, min upright {min(up):.4f}, "
              f"radius {rad[0]:.3f} -> {rad[-1]:.3f} m, moved {np.hypot(fr[-1]['root_pos'][0] - fr[0]['root_pos'][0], fr[-1]['root_pos'][1] - fr[0]['root_pos'][1]):.3f} m")
        pi = prefixes.index(p)
        out[p] = dict(scene=scene, prefix=p, command=f32(cmd), goal=goals[pi], goal_stop_dist=stops[pi], goal_vmax=vmaxs[pi],
                      ctrl_dt=spec["ctrl_dt"], decimation=spec["decimation"], obs_dim=len(fr[0]["obs"]), frames=fr)
    if len(prefixes) == 2:
        a, b = (R[p]["frames"][-1]["root_pos"] for p in prefixes); print(f"final separation {np.hypot(a[0] - b[0], a[1] - b[1]):.3f} m")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--policy-rung", default=None, help="reuse <policy-rung>_policy.onnx instead of exporting")
    ap.add_argument("--scene", default="scene_flat_1p_train.xml"); ap.add_argument("--robots", nargs="+", default=["a_"])
    ap.add_argument("--cmd", type=float, nargs=3, default=[0, 0, 0]); ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--goal", default=["none"], nargs="+", choices=["none", "center", "opponent"]); ap.add_argument("--stop-dist", type=float, nargs="+", default=[0.3])
    ap.add_argument("--vmax", type=float, nargs="+", default=[0.8]); ap.add_argument("--policies", nargs="+", default=None, help="existing <name>_policy.onnx per robot (skips export)")
    ap.add_argument("--spawn", type=float, nargs="+", default=None, help="r bearing yaw per robot (3 or 6 numbers)")
    a = ap.parse_args()
    os.makedirs(UNITY_MODELS, exist_ok=True)
    onnx_path = os.path.join(UNITY_MODELS, f"{a.rung}_policy.onnx")
    if a.ckpt: export_onnx(a.ckpt, onnx_path)
    elif a.policy_rung: shutil.copyfile(os.path.join(UNITY_MODELS, f"{a.policy_rung}_policy.onnx"), onnx_path)
    spawns = {p: a.spawn[3 * i:3 * i + 3] for i, p in enumerate(a.robots)} if a.spawn else None
    one = lambda v: v[0] if len(v) == 1 else v
    paths = [os.path.join(UNITY_MODELS, f"{n}_policy.onnx") for n in a.policies] if a.policies else onnx_path
    refs = record(paths, a.scene, a.robots, a.cmd, a.seconds, one(a.goal), spawns, one(a.stop_dist), one(a.vmax))
    for p, ref in refs.items():
        suffix = "" if len(refs) == 1 else "_" + p.strip("_")
        out = os.path.join(UNITY_MODELS, f"{a.rung}_reference_trajectory{suffix}.json")
        json.dump(ref, open(out, "w")); print("wrote", out)


if __name__ == "__main__":
    main()
