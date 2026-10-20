"""Write the weights of Unity's ONNX brains as small binary files for the Tactician's rollouts (Unity's MlpNet.cs runs
them on worker threads, where Sentis workers cannot run).

  uv run python scripts/export_mlp.py                       # the game roster's fighting brains and their critics (ROSTER)
  uv run python scripts/export_mlp.py --onnx ../Assets/PoKingHill/Models/attacker_policy.onnx

<name>_policy.onnx -> <name>_mlp.bytes, <name>_value.onnx -> <name>_value_mlp.bytes, next to the ONNX file. Layout (little
endian): int32 magic 0x504C4D4B ("KMLP"), int32 version 1, int32 L (layers), int32 dims[L + 1], float32 mean[dims[0]],
float32 divisor[dims[0]] (std + 1e-2), then per layer float32 W[out x in] (row-major) and float32 b[out]. The forward pass
is x = (obs - mean) / divisor, then Linear/ELU, the last layer linear: the same as koth/actor.py. Each file is checked
against onnxruntime on 1,000 random inputs with a numpy implementation of that forward pass."""
import argparse, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, onnxruntime as ort
from scripts.onnx_to_ckpt import convert
from koth.actor import NORM_EPS, mlp_indices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.normpath(os.path.join(ROOT, "..", "Assets", "PoKingHill", "Models"))
# the fighting brains ParityBatch.BuildDuelDemo seats (GoalMode.Opponent) and the critics next to them, relative to MODELS
ROSTER = ["attacker", "duelist", "rammer", "ffa", "kim/fighter", "trump/fighter"]


def layers(sd: dict):
    mean = sd["obs_normalizer._mean"].numpy().reshape(-1).astype(np.float32)
    div = (sd["obs_normalizer._std"].numpy().reshape(-1) + NORM_EPS).astype(np.float32)
    ws = [(sd[f"mlp.{i}.weight"].numpy().astype(np.float32), sd[f"mlp.{i}.bias"].numpy().astype(np.float32)) for i in mlp_indices(sd)]
    return mean, div, ws


def forward(mean, div, ws, x):
    h = (x - mean) / div
    for k, (w, b) in enumerate(ws):
        h = h @ w.T + b
        if k < len(ws) - 1: h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
    return h


def write(onnx_path: str) -> str:
    stem = os.path.basename(onnx_path)[:-len(".onnx")]
    out = os.path.join(os.path.dirname(onnx_path), (stem[:-len("_policy")] + "_mlp" if stem.endswith("_policy") else stem + "_mlp") + ".bytes")
    mean, div, ws = layers(convert(onnx_path)["actor_state_dict"])
    dims = [ws[0][0].shape[1]] + [w.shape[0] for w, _ in ws]
    with open(out, "wb") as f:
        f.write(struct.pack("<iii", 0x504C4D4B, 1, len(ws))); f.write(struct.pack(f"<{len(dims)}i", *dims))
        f.write(mean.tobytes()); f.write(div.tobytes())
        for w, b in ws: f.write(np.ascontiguousarray(w).tobytes()); f.write(b.tobytes())
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]); name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((1000, dims[0])).astype(np.float32) * 2
    ref = np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])
    err = float(np.abs(forward(mean, div, ws, x) - ref).max()); scale = max(1.0, float(np.abs(ref).max()))
    assert err < 1e-4 * scale, f"{out}: numpy forward differs from the ONNX by {err:.2e}"
    print(f"wrote {out}: dims {dims}, {os.path.getsize(out) / 1024:.0f} KB, max error vs ONNX {err:.1e}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--onnx", nargs="*", default=None); a = ap.parse_args()
    paths = a.onnx or [p for n in ROSTER for p in (os.path.join(MODELS, n + "_policy.onnx"), os.path.join(MODELS, n + "_value.onnx")) if os.path.exists(p)]
    for p in paths: write(p)
