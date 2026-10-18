"""Rebuild an rsl_rl-style checkpoint from an exported Unity policy (ONNX), for machines that have the brains but not
training/runs. The export is (obs - mean) / divisor followed by Linear/ELU layers, where divisor = std + 1e-2, so the
actor is recovered exactly; there is no critic (warm starts train a fresh one).

    uv run python scripts/onnx_to_ckpt.py --onnx ../Assets/PoKingHill/Models/attacker_policy.onnx --out runs/league/careful1.pt [--std 0.3]

The checkpoint holds actor_state_dict (obs_normalizer _mean/_var/_std/count, mlp.0/2/4/6, distribution.std_param),
an empty critic_state_dict and the source path. The normalizer count is set high so that a warm-started run keeps
the trained statistics instead of re-estimating them from its first batch. Checked: torch actor vs onnxruntime on
1,000 random inputs (max abs error printed; fails above 1e-4)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import numpy_helper
from koth.actor import Actor, NORM_EPS


def convert(onnx_path: str, std: float = 0.3) -> dict:
    model = onnx.load(onnx_path)
    inits = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()).float() for i in model.graph.initializer}
    mean = next(v for k, v in inits.items() if k.endswith("mean"))
    div = next(v for k, v in inits.items() if k.startswith("onnx::Div"))
    # the Gemm nodes in graph order give the layer order; their B input is the weight, C the bias (transB = 1)
    gemms = [n for n in model.graph.node if n.op_type == "Gemm"]
    sd = {"obs_normalizer._mean": mean, "obs_normalizer._std": div - NORM_EPS, "obs_normalizer._var": (div - NORM_EPS) ** 2,
          "obs_normalizer.count": torch.tensor(10 ** 9, dtype=torch.long)}
    for k, g in enumerate(gemms):
        sd[f"mlp.{2 * k}.weight"] = inits[g.input[1]]; sd[f"mlp.{2 * k}.bias"] = inits[g.input[2]]
    sd["distribution.std_param"] = torch.full((sd[f"mlp.{2 * (len(gemms) - 1)}.weight"].shape[0],), std)
    return {"actor_state_dict": sd, "critic_state_dict": {}, "iter": 0, "infos": {"from_onnx": os.path.abspath(onnx_path)}}


def check(ck: dict, onnx_path: str) -> float:
    actor = Actor(ck["actor_state_dict"]).eval(); dim = actor.obs_dim
    x = np.random.default_rng(0).standard_normal((1000, dim)).astype(np.float32) * 2
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]); name = sess.get_inputs()[0].name
    ref = np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])
    with torch.no_grad(): got = actor(torch.from_numpy(x)).numpy()
    return float(np.abs(got - ref).max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--onnx", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--std", type=float, default=0.3, help="exploration std stored for stochastic frozen opponents and warm starts")
    a = ap.parse_args()
    ck = convert(a.onnx, a.std); err = check(ck, a.onnx)
    if err > 1e-4: sys.exit(f"{a.onnx}: rebuilt actor differs from the ONNX by {err:.2e}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); torch.save(ck, a.out)
    print(f"{a.out}: {ck['actor_state_dict']['obs_normalizer._mean'].shape[1]} inputs, max error vs ONNX {err:.2e}")
