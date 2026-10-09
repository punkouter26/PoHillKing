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
import numpy as np, torch, mujoco, onnx, onnxruntime as ort
from koth.obs import build_obs, build_combat, goal_command, OBS_DIM, GAIT_FREQ_HZ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", os.environ.get("KOTH_ROBOT", "g1"))
UNITY_MODELS = os.path.normpath(os.path.join(ROOT, "..", "Assets", "PoKingHill", "Models"))
SLOPE_K, PLATEAU_R = 1.0 / 9.0, 1.5


def export_onnx(ckpt: str, out_path: str) -> None:
    from rsl_rl.runners import OnPolicyRunner
    from koth.env import KothEnv, default_cfg
    from scripts.train import train_cfg
    env = KothEnv(default_cfg("r0"), 16)
    runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cuda")
    runner.load(ckpt)
    net = runner.alg.get_policy().as_onnx(verbose=False).to("cpu").eval()
    torch.onnx.export(net, (torch.zeros(1, OBS_DIM),), out_path, export_params=True, opset_version=17,
                      input_names=["obs"], output_names=["actions"], dynamo=False)
    m = onnx.load(out_path); onnx.checker.check_model(m)
    onnx.save(m, out_path, save_as_external_data=False)             # one self-contained file for Unity
    if os.path.exists(out_path + ".data"): os.remove(out_path + ".data")
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = torch.randn(1000, OBS_DIM) * 2
    with torch.no_grad():
        ref = net(x).numpy()
    got = np.concatenate([sess.run(None, {"obs": x[i:i + 1].numpy()})[0] for i in range(1000)])
    err = float(np.abs(ref - got).max())
    print(f"onnx: opset {m.opset_import[0].version}, input {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}, "
          f"torch-vs-onnx max abs err {err:.2e}")
    assert err < 1e-5, "ONNX export does not reproduce the torch policy"


def record(onnx_path, scene, prefixes, cmd, seconds, goal="none", spawns=None, stop_dist=0.3, vmax=0.8) -> dict:
    # onnx_path, goal, stop_dist and vmax may each be one value or one per robot (a duel has two different brains)
    per = lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * len(prefixes)
    onnx_paths, goals, stops, vmaxs = per(onnx_path), per(goal), per(stop_dist), per(vmax)
    """Returns {prefix: reference dict}. spawns: {prefix: (r, bearing, yaw)} or None for the keyframe pose."""
    spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, scene)); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
    R = {}
    for p in prefixes:
        jid = [mujoco.mj_name2id(m, J, p + j) for j in spec["joints"]]
        root = mujoco.mj_name2id(m, J, p + "floating_base_joint")
        aid = np.array([mujoco.mj_name2id(m, A, p + j) for j in spec["joints"]])
        R[p] = dict(qadr=np.array([m.jnt_qposadr[j] for j in jid]), dadr=np.array([m.jnt_dofadr[j] for j in jid]), aid=aid,
                    rq=m.jnt_qposadr[root], rd=m.jnt_dofadr[root], lo=m.actuator_ctrlrange[aid, 0], hi=m.actuator_ctrlrange[aid, 1],
                    last=np.zeros(29, dtype=np.float32), phase=np.float32(0.0), frames=[], cmd=np.array(cmd, dtype=np.float32), ctrl=None)
        if spawns and p in spawns:               # same placement rule as KothEnv.reset_idx
            r, bearing, yaw = spawns[p]; rq = R[p]["rq"]
            slope = 2 * SLOPE_K * max(r - PLATEAU_R, 0.0); h = -SLOPE_K * max(r - PLATEAU_R, 0.0) ** 2
            d.qpos[rq:rq + 3] = [r * np.cos(bearing), r * np.sin(bearing), spec["key_root_z"] + h + 0.02 + 0.12 * slope]
            d.qpos[rq + 3:rq + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    pool = [(m.jnt_qposadr[mujoco.mj_name2id(m, J, b + "_free")], m.jnt_dofadr[mujoco.mj_name2id(m, J, b + "_free")]) for b in spec["pool_bodies"]]
    park = [d.qpos[q:q + 7].copy() for q, _ in pool]
    default = np.array(spec["default_pose"], dtype=np.float32)
    sessions = [ort.InferenceSession(p_, providers=["CPUExecutionProvider"]) for p_ in onnx_paths]
    f32 = lambda a: [float(np.float32(v)) for v in a]; f64 = lambda a: [float(v) for v in a]
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float64))[None]
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
            obs = build_obs(T(d.qpos[rq + 3:rq + 7]), T(d.qvel[rd:rd + 3]), T(d.qvel[rd + 3:rd + 6]), T(d.qpos[s["qadr"]]), T(d.qvel[s["dadr"]]),
                            torch.tensor(default, dtype=torch.float64), T(s["last"]), T(s["cmd"]),
                            torch.tensor([float(s["phase"])], dtype=torch.float64))
            if sess.get_inputs()[0].shape[1] > OBS_DIM:      # combat policy: append the opponent/ring block
                oq, od = other["rq"], other["rd"]
                obs = torch.cat([obs, build_combat(T(d.qpos[rq:rq + 3]), T(d.qpos[rq + 3:rq + 7]), T(d.qvel[rd:rd + 3]),
                                                   T(d.qpos[oq:oq + 3]), T(d.qpos[oq + 3:oq + 7]), T(d.qvel[od:od + 3]))], 1)
            obs = obs[0].numpy().astype(np.float32)
            raw = sess.run(None, {"obs": obs[None]})[0][0]
            act = np.clip(raw, -1, 1).astype(np.float32)
            s["ctrl"] = np.clip(default.astype(np.float64) + spec["action_scale"] * act.astype(np.float64), s["lo"], s["hi"])
            s["frames"].append(dict(t=float(d.time), root_pos=f64(d.qpos[rq:rq + 3]), root_quat=f64(d.qpos[rq + 3:rq + 7]),
                                    root_linvel=f64(d.qvel[rd:rd + 3]), root_angvel=f64(d.qvel[rd + 3:rd + 6]),
                                    joint_pos=f64(d.qpos[s["qadr"]]), joint_vel=f64(d.qvel[s["dadr"]]),
                                    obs=f32(obs), action=f32(raw), ctrl=f64(s["ctrl"]), command=f32(s["cmd"])))
            s["last"] = act; s["phase"] = np.float32(s["phase"] + np.float32(2 * np.pi * GAIT_FREQ_HZ * spec["ctrl_dt"]))
            if s["phase"] > 2 * np.pi: s["phase"] = np.float32(s["phase"] - np.float32(2 * np.pi))
        for _ in range(spec["decimation"]):
            for (q, dof), pk in zip(pool, park):
                d.qpos[q:q + 7] = pk; d.qvel[dof:dof + 6] = 0
            for p in prefixes: d.ctrl[R[p]["aid"]] = R[p]["ctrl"]
            mujoco.mj_step(m, d)
    out = {}
    for p in prefixes:
        fr = R[p]["frames"]; up = [1 - 2 * (f["root_quat"][1] ** 2 + f["root_quat"][2] ** 2) for f in fr]
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
