"""Export any checkpoint's actor to a single-file ONNX for Unity (opset 17, batch 1), whatever its input size.
  uv run python scripts/export_policy.py --ckpt runs/r4att/r4att_c/model_500.pt --name attacker [--value]
Writes Assets/PoKingHill/Models[/<robot>]/<name>_policy.onnx and checks ONNX against torch on 1000 random inputs.
--value also writes <name>_value.onnx: the critic (obs normalizer folded in, input = the critic observation, output
"value" (1, 1)), for a win-probability style bar in Unity. Its input layout is the env's critic observation:
[103 clean locomotion obs, world linear velocity (3), pelvis height above ground (1), feet contact (2),
DR params (mass scale, friction, kp scale) (3)] + the 9 combat inputs for combat brains (112 or 121 wide)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, onnx, onnxruntime as ort
from koth.actor import Actor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_module(net: torch.nn.Module, dim: int, path: str, output_name: str = "actions") -> float:
    """torch.onnx export (opset 17, batch 1, input "obs"), re-saved as one self-contained file. Returns the max abs
    torch-vs-onnxruntime error over 1000 random inputs (std 2)."""
    torch.onnx.export(net, (torch.zeros(1, dim),), path, export_params=True, opset_version=17, input_names=["obs"], output_names=[output_name], dynamo=False)
    m = onnx.load(path); onnx.checker.check_model(m); onnx.save(m, path, save_as_external_data=False)
    if os.path.exists(path + ".data"): os.remove(path + ".data")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); x = torch.randn(1000, dim) * 2
    with torch.no_grad(): ref = net(x).numpy()
    return float(np.abs(ref - np.concatenate([sess.run(None, {"obs": x[i:i + 1].numpy()})[0] for i in range(1000)])).max())


def export(ckpt: str, path: str, which: str = "actor") -> int:
    """Checkpoint actor (which="actor") or critic (which="critic") -> ONNX at path. Returns the input width."""
    net = Actor(torch.load(ckpt, map_location="cpu", weights_only=False)[f"{which}_state_dict"], torch.float32).eval()
    dim = net.obs_dim; os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    err = export_module(net, dim, path, "actions" if which == "actor" else "value")
    print(f"wrote {path}: input {dim}, torch-vs-onnx max abs err {err:.2e}")
    assert err < 1e-5
    return dim


if __name__ == "__main__":
    from koth.build_mjcf import UNITY_MODELS          # Models/ for the G1, Models/<robot>/ otherwise (KOTH_ROBOT)
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--value", action="store_true", help="also export the critic to <name>_value.onnx")
    ap.add_argument("--out-dir", default=UNITY_MODELS, help="output folder (default: the Unity Models folder)")
    a = ap.parse_args()
    export(a.ckpt, os.path.join(a.out_dir, f"{a.name}_policy.onnx"))
    if a.value: export(a.ckpt, os.path.join(a.out_dir, f"{a.name}_value.onnx"), which="critic")
