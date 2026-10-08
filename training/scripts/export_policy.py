"""Export any checkpoint's actor to a single-file ONNX for Unity (opset 17, batch 1), whatever its input size.
  uv run python scripts/export_policy.py --ckpt runs/r4att/r4att_c/model_500.pt --name attacker
Writes Assets/PoKingHill/Models/<name>_policy.onnx and checks ONNX against torch on 1000 random inputs."""
import argparse, os, sys
import numpy as np, torch, onnx, onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.normpath(os.path.join(ROOT, "..", "Assets", "PoKingHill", "Models"))


class Actor(torch.nn.Module):
    """Obs normalizer + ELU MLP rebuilt from an rsl_rl actor state dict (deterministic mean action)."""
    def __init__(self, sd):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].float()); self.register_buffer("std", sd["obs_normalizer._std"].float())
        idx = sorted(int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight"))
        self.layers = torch.nn.ModuleList()
        for i in idx:
            w = sd[f"mlp.{i}.weight"]; lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data = w.float(); lin.bias.data = sd[f"mlp.{i}.bias"].float(); self.layers.append(lin)

    def forward(self, obs):
        x = (obs - self.mean) / (self.std + 1e-2)
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1: x = torch.nn.functional.elu(x)
        return x


ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True); a = ap.parse_args()
net = Actor(torch.load(a.ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]).eval()
dim = net.mean.shape[1]; os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"{a.name}_policy.onnx")
torch.onnx.export(net, (torch.zeros(1, dim),), path, export_params=True, opset_version=17, input_names=["obs"], output_names=["actions"], dynamo=False)
m = onnx.load(path); onnx.checker.check_model(m); onnx.save(m, path, save_as_external_data=False)
if os.path.exists(path + ".data"): os.remove(path + ".data")
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); x = torch.randn(1000, dim) * 2
with torch.no_grad(): ref = net(x).numpy()
err = float(np.abs(ref - np.concatenate([sess.run(None, {"obs": x[i:i + 1].numpy()})[0] for i in range(1000)])).max())
print(f"wrote {path}: input {dim}, torch-vs-onnx max abs err {err:.2e}")
assert err < 1e-5
