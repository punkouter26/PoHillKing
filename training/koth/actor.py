"""Deterministic actor (or critic) rebuilt from an rsl_rl model state dict: observation normalizer + ELU MLP.

Torch only (no warp import), so the CPU scripts and the ONNX exporter can use it. Same arithmetic as rsl_rl's
MLPModel in eval mode: x = (obs - mean) / (std + 1e-2), then Linear/ELU layers, the last one without activation.
Used by env.FrozenPolicy (GPU, float32), scripts/play.py (CPU, float64) and scripts/export_policy.py (ONNX)."""
from __future__ import annotations
import torch

NORM_EPS = 1e-2          # rsl_rl EmpiricalNormalization eps


def mlp_indices(sd: dict) -> list[int]:
    return sorted(int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight"))


class Actor(torch.nn.Module):
    """sd = checkpoint["actor_state_dict"] (or ["critic_state_dict"]). dtype None keeps the stored dtype and device.
    Construction draws no random numbers (layers are made with skip_init), so it can run inside a seeded env."""
    def __init__(self, sd: dict, dtype: torch.dtype | None = None):
        super().__init__()
        conv = (lambda t: t) if dtype is None else (lambda t: t.to(dtype))
        self.register_buffer("mean", conv(sd["obs_normalizer._mean"])); self.register_buffer("std", conv(sd["obs_normalizer._std"]))
        self.layers = torch.nn.ModuleList()
        for i in mlp_indices(sd):
            w = conv(sd[f"mlp.{i}.weight"])
            lin = torch.nn.utils.skip_init(torch.nn.Linear, w.shape[1], w.shape[0], device=w.device, dtype=w.dtype)
            lin.weight.data = w; lin.bias.data = conv(sd[f"mlp.{i}.bias"]); self.layers.append(lin)

    @property
    def obs_dim(self) -> int:
        return self.mean.shape[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / (self.std + NORM_EPS)
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1: x = torch.nn.functional.elu(x)
        return x


def load(ckpt: str, which: str = "actor", device="cpu", dtype: torch.dtype | None = None) -> Actor:
    """Actor (or critic, which="critic") of an rsl_rl checkpoint file, in eval mode."""
    sd = torch.load(ckpt, map_location=device, weights_only=False)[f"{which}_state_dict"]
    return Actor(sd, dtype).eval()
