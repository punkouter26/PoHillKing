"""Export any checkpoint's actor to a single-file ONNX for Unity (opset 17, batch 1), whatever its input size.
  uv run python scripts/export_policy.py --ckpt runs/r4att/r4att_c/model_500.pt --name attacker [--value]
Writes Assets/PoKingHill/Models[/<robot>]/<name>_policy.onnx and checks ONNX against torch on 1000 random inputs.
--value also writes <name>_value.onnx: the critic (obs normalizer folded in, input = the critic observation, output
"value" (1, 1)), for a win-probability style bar in Unity. Its input layout is the env's critic observation:
[103 clean locomotion obs, world linear velocity (3), pelvis height above ground (1), feet contact (2),
DR params (mass scale, friction, kp scale) (3)] + the 9 combat inputs for combat brains (112 or 121 wide).

A brain trained with a changed action path (koth/action_map.py: tanh map, per-joint scale, torque window) gets
<name>_action.json beside the ONNX, which Unity's PolicyRunner and duel_stats.py read; exporting a brain with the
original path removes a stale one. A checkpoint whose run wrote graduation.json / manifest.json with a failed bar (a NaN
gradient, or a --graduate-reward target not reached) is refused unless --force: it has not graduated (AGENTS.md)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, onnx
from koth import cpu_sim
from koth.actor import Actor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_module(net: torch.nn.Module, dim: int, path: str, output_name: str = "actions") -> float:
    """torch.onnx export (opset 17, batch 1, input "obs"), re-saved as one self-contained file. Returns the max abs
    torch-vs-onnxruntime error over 1000 random inputs (std 2)."""
    torch.onnx.export(net, (torch.zeros(1, dim),), path, export_params=True, opset_version=17, input_names=["obs"], output_names=[output_name], dynamo=False)
    m = onnx.load(path); onnx.checker.check_model(m); onnx.save(m, path, save_as_external_data=False)
    if os.path.exists(path + ".data"): os.remove(path + ".data")
    x = torch.randn(1000, dim) * 2
    with torch.no_grad(): ref = net(x).numpy()
    return float(np.abs(ref - cpu_sim.onnx_rows(cpu_sim.session(path), x.numpy())).max())


def graduation_problem(ckpt: str) -> str | None:
    """Why the run that wrote ckpt has not graduated (None = it has, or there is no record: league files, rebuilt ONNX)."""
    d = os.path.dirname(os.path.abspath(ckpt))
    for f in ("graduation.json", "manifest.json"):
        p = os.path.join(d, f)
        if not os.path.exists(p): continue
        import json
        g = json.load(open(p)); g = g.get("bars", g) if f == "manifest.json" else g
        if not g: continue
        if g.get("updates_with_nan_gradients", 0): return f"{g['updates_with_nan_gradients']} PPO updates had NaN/inf gradients ({f})"
        if g.get("target_mean_reward_100_episodes") is not None and not g.get("graduated"):
            return f"last-100-episode mean reward {g.get('final_mean_reward_100_episodes')} below the target {g['target_mean_reward_100_episodes']} ({f})"
        return None
    return None


def export(ckpt: str, path: str, which: str = "actor") -> int:
    """Checkpoint actor (which="actor") or critic (which="critic") -> ONNX at path. Returns the input width."""
    net = Actor(torch.load(ckpt, map_location="cpu", weights_only=False)[f"{which}_state_dict"], torch.float32).eval()
    dim = net.obs_dim; os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    err = export_module(net, dim, path, "actions" if which == "actor" else "value")
    print(f"wrote {path}: input {dim}, torch-vs-onnx max abs err {err:.2e}")
    # actions are O(1): 1e-5 absolute; a critic's returns are O(10-100), so its bound is relative to its output size
    scale = 1.0 if which == "actor" else max(1.0, float(net(torch.randn(256, dim) * 2).abs().max()))
    assert err < 1e-5 * scale, f"{path}: torch-vs-onnx error {err:.2e} above {1e-5 * scale:.2e}"
    return dim


if __name__ == "__main__":
    from koth.build_mjcf import UNITY_MODELS          # Models/ for the G1, Models/<robot>/ otherwise (KOTH_ROBOT)
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--value", action="store_true", help="also export the critic to <name>_value.onnx")
    ap.add_argument("--out-dir", default=UNITY_MODELS, help="output folder (default: the Unity Models folder)")
    ap.add_argument("--force", action="store_true", help="export even though the run did not graduate")
    a = ap.parse_args()
    why = graduation_problem(a.ckpt)
    if why and not a.force: sys.exit(f"refused: {a.ckpt} has not graduated: {why} (--force to export anyway)")
    if why: print(f"warning: exporting a brain that has not graduated: {why}")
    out = os.path.join(a.out_dir, f"{a.name}_policy.onnx")
    export(a.ckpt, out)
    from koth.action_map import ckpt_spec, write_sidecar
    side = write_sidecar(out, ckpt_spec(a.ckpt))
    if side: print("wrote", side, json.dumps(ckpt_spec(a.ckpt))[:160])
    if a.value: export(a.ckpt, os.path.join(a.out_dir, f"{a.name}_value.onnx"), which="critic")
