"""Export a checkpoint to a single-file ONNX (opset 17, batch 1) and record the parity reference trajectory.

  uv run python scripts/record_reference.py --rung r0 --ckpt runs/r0/r0_c/model_final.pt [--cmd 0 0 0] [--seconds 5]

The reference is produced by CPU MuJoCo (double precision, the same engine the Unity plugin runs) driven by the
exported ONNX through onnxruntime, with the exact tick order of Unity's PolicyRunner: build obs, infer, write
ctrl, then 10 physics steps. No noise, no domain randomization, no pushes; pool boxes pinned every physics step.
Outputs into Assets/PoKingHill/Models/:  <rung>_policy.onnx   <rung>_reference_trajectory.json
Checks: torch policy vs ONNX max abs error < 1e-5 on 1000 random observations."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, mujoco, onnx, onnxruntime as ort
from koth.obs import build_obs, goal_command, OBS_DIM, GAIT_FREQ_HZ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "g1")
UNITY_MODELS = os.path.normpath(os.path.join(ROOT, "..", "Assets", "PoKingHill", "Models"))


def export_onnx(ckpt: str, out_path: str) -> None:
    from rsl_rl.runners import OnPolicyRunner
    from koth.env import KothEnv, default_cfg
    from scripts.train import train_cfg
    env = KothEnv(default_cfg("r0"), 16)
    runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cuda")
    runner.load(ckpt)
    policy = runner.alg.get_policy()
    net = policy.as_onnx(verbose=False).to("cpu").eval()
    torch.onnx.export(net, (torch.zeros(1, OBS_DIM),), out_path, export_params=True, opset_version=17,
                      input_names=["obs"], output_names=["actions"], dynamo=False)
    m = onnx.load(out_path); onnx.checker.check_model(m)
    onnx.save(m, out_path, save_as_external_data=False)             # one self-contained file for Unity
    for f in (out_path + ".data",):
        if os.path.exists(f): os.remove(f)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = torch.randn(1000, OBS_DIM) * 2
    with torch.no_grad():
        ref = net(x).numpy()
    got = np.concatenate([sess.run(None, {"obs": x[i:i + 1].numpy()})[0] for i in range(1000)])
    err = float(np.abs(ref - got).max())
    print(f"onnx: opset {m.opset_import[0].version}, input {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}, "
          f"torch-vs-onnx max abs err {err:.2e}")
    assert err < 1e-5, "ONNX export does not reproduce the torch policy"


def record(onnx_path: str, scene: str, prefix: str, cmd, seconds: float, goal: str = "none", spawn=None, stop_dist: float = 0.3, vmax: float = 0.8) -> dict:
    spec = json.load(open(os.path.join(ASSETS, "joint_map.json")))
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, scene)); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
    jid = [mujoco.mj_name2id(m, J, prefix + j) for j in spec["joints"]]
    qadr = np.array([m.jnt_qposadr[j] for j in jid]); dadr = np.array([m.jnt_dofadr[j] for j in jid])
    aid = np.array([mujoco.mj_name2id(m, A, prefix + j) for j in spec["joints"]])
    root = mujoco.mj_name2id(m, J, prefix + "floating_base_joint"); rq, rd = m.jnt_qposadr[root], m.jnt_dofadr[root]
    pool = [(m.jnt_qposadr[mujoco.mj_name2id(m, J, b + "_free")], m.jnt_dofadr[mujoco.mj_name2id(m, J, b + "_free")]) for b in spec["pool_bodies"]]
    park = [d.qpos[q:q + 7].copy() for q, _ in pool]
    default = np.array(spec["default_pose"], dtype=np.float32)
    lo, hi = m.actuator_ctrlrange[aid, 0], m.actuator_ctrlrange[aid, 1]
    if spawn is not None:                      # r, bearing, yaw: same placement rule as KothEnv.reset_idx
        r, bearing, yaw = spawn
        slope = 2 * (1.0 / 9.0) * max(r - 1.5, 0.0); h = -(1.0 / 9.0) * max(r - 1.5, 0.0) ** 2
        d.qpos[rq:rq + 3] = [r * np.cos(bearing), r * np.sin(bearing), spec["key_root_z"] + h + 0.02 + 0.12 * slope]
        d.qpos[rq + 3:rq + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    last = np.zeros(29, dtype=np.float32); phase = 0.0; frames = []
    cmd = np.array(cmd, dtype=np.float32)
    f32 = lambda a: [float(np.float32(v)) for v in a]
    for tick in range(int(round(seconds / spec["ctrl_dt"]))):
        if goal == "center":
            cmd = goal_command(torch.tensor(d.qpos[rq + 3:rq + 7])[None], torch.tensor([[-d.qpos[rq], -d.qpos[rq + 1]]]), stop_dist, vmax)[0].numpy().astype(np.float32)
        obs = build_obs(torch.tensor(d.qpos[rq + 3:rq + 7], dtype=torch.float64)[None], torch.tensor(d.qvel[rd:rd + 3])[None],
                        torch.tensor(d.qvel[rd + 3:rd + 6])[None], torch.tensor(d.qpos[qadr])[None], torch.tensor(d.qvel[dadr])[None],
                        torch.tensor(default, dtype=torch.float64), torch.tensor(last, dtype=torch.float64)[None],
                        torch.tensor(cmd, dtype=torch.float64)[None], torch.tensor([phase], dtype=torch.float64))[0].numpy().astype(np.float32)
        raw = sess.run(None, {"obs": obs[None]})[0][0]
        act = np.clip(raw, -1, 1).astype(np.float32)
        ctrl = np.clip(default.astype(np.float64) + spec["action_scale"] * act.astype(np.float64), lo, hi)
        frames.append(dict(t=float(d.time), root_pos=[float(v) for v in d.qpos[rq:rq + 3]], root_quat=[float(v) for v in d.qpos[rq + 3:rq + 7]],
                           root_linvel=[float(v) for v in d.qvel[rd:rd + 3]], root_angvel=[float(v) for v in d.qvel[rd + 3:rd + 6]],
                           joint_pos=[float(v) for v in d.qpos[qadr]], joint_vel=[float(v) for v in d.qvel[dadr]],
                           obs=f32(obs), action=f32(raw), ctrl=[float(v) for v in ctrl]))
        last = act; phase = np.float32(phase + 2 * np.pi * GAIT_FREQ_HZ * spec["ctrl_dt"])
        if phase > 2 * np.pi: phase = np.float32(phase - 2 * np.pi)
        for _ in range(spec["decimation"]):
            for (q, dof), p in zip(pool, park):
                d.qpos[q:q + 7] = p; d.qvel[dof:dof + 6] = 0
            d.ctrl[aid] = ctrl
            mujoco.mj_step(m, d)
    up = [1 - 2 * (f["root_quat"][1] ** 2 + f["root_quat"][2] ** 2) for f in frames]
    print(f"reference: {len(frames)} ticks, pelvis z {frames[0]['root_pos'][2]:.4f} -> {frames[-1]['root_pos'][2]:.4f} "
          f"(min {min(f['root_pos'][2] for f in frames):.4f}), min upright {min(up):.4f}, "
          f"final xy drift {np.hypot(frames[-1]['root_pos'][0] - frames[0]['root_pos'][0], frames[-1]['root_pos'][1] - frames[0]['root_pos'][1]):.3f} m")
    print(f"final radius {float(np.hypot(*frames[-1]['root_pos'][:2])):.3f} m")
    return dict(scene=scene, prefix=prefix, command=[float(c) for c in cmd], goal=goal, goal_stop_dist=stop_dist, goal_vmax=vmax, ctrl_dt=spec["ctrl_dt"], decimation=spec["decimation"],
                obs_dim=OBS_DIM, frames=frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--scene", default="scene_flat_1p_train.xml"); ap.add_argument("--prefix", default="a_")
    ap.add_argument("--cmd", type=float, nargs=3, default=[0, 0, 0]); ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--goal", default="none", choices=["none", "center"]); ap.add_argument("--spawn", type=float, nargs=3, default=None, help="r bearing yaw")
    ap.add_argument("--policy-rung", default=None, help="reuse <policy-rung>_policy.onnx instead of exporting")
    a = ap.parse_args()
    os.makedirs(UNITY_MODELS, exist_ok=True)
    onnx_path = os.path.join(UNITY_MODELS, f"{a.rung}_policy.onnx")
    if a.ckpt: export_onnx(a.ckpt, onnx_path)
    if a.policy_rung and not a.ckpt:
        import shutil; shutil.copyfile(os.path.join(UNITY_MODELS, f"{a.policy_rung}_policy.onnx"), onnx_path)
    ref = record(onnx_path, a.scene, a.prefix, a.cmd, a.seconds, a.goal, a.spawn)
    out = os.path.join(UNITY_MODELS, f"{a.rung}_reference_trajectory.json")
    json.dump(ref, open(out, "w"))
    print("wrote", onnx_path, "and", out)


if __name__ == "__main__":
    main()
