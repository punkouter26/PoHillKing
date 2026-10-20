"""How a brain's output becomes a joint target. Mirrored by Unity's ActionMap.cs; the spec travels with an exported brain
as <name>_action.json beside its ONNX (scripts/export_policy.py). No file = the original path, which every brain up to
2026-10-20 uses:
  a      = clip(raw, -1, 1)                       (also the "last action" observation)
  target = clip(default + style offset + 0.5 * a, ctrl range)

Options (env cfg "action", all off by default):
  map "tanh"    a = tanh(raw) instead of the clip. The clip has no gradient beyond +-1: the champion's outputs sat on it
                29 % of the time (drive_audit.py, 2026-10-20), so exploration noise there changed nothing and the mean
                could drift outward unopposed. tanh keeps every output inside (-1, 1) and every change of it effective.
  scale [29]    per-joint action scale (rad per unit action) instead of joint_map's single 0.5.
  window True   torque-feasible targets: the target is clamped to what the actuator can act on at the current state,
                kp (target - q) - kv qd within +-forcerange, i.e. target in [q + (kv qd - tau) / kp, q + (kv qd + tau) / kp]
                with the nominal kp, kv, tau. A target beyond the window asks for torque the motor does not have; it
                would only wind up (a full-torque push that stays on after the joint has passed where it should stop).
The style low-pass (shove) filters a, before the scale, exactly as before."""
from __future__ import annotations
import json, os
import numpy as np

DEFAULT = dict(map="clip", scale=None, window=False)


def spec_of(cfg_action: dict | None) -> dict:
    """Complete spec from an env cfg "action" entry (None / missing keys = the original path)."""
    s = dict(DEFAULT); s.update(cfg_action or {})
    assert s["map"] in ("clip", "tanh"), s["map"]
    if s["scale"] is not None: s["scale"] = [float(x) for x in s["scale"]]; assert len(s["scale"]) == 29
    return s


def is_default(s: dict) -> bool:
    s = spec_of(s); return s["map"] == "clip" and s["scale"] is None and not s["window"]


def sidecar_path(onnx_path: str) -> str:
    return onnx_path[:-len("_policy.onnx")] + "_action.json" if onnx_path.endswith("_policy.onnx") else onnx_path + ".action.json"


def write_sidecar(onnx_path: str, s: dict) -> str | None:
    """<name>_action.json beside <name>_policy.onnx, or remove a stale one when the brain uses the original path."""
    p = sidecar_path(onnx_path); s = spec_of(s)
    if is_default(s):
        if os.path.exists(p): os.remove(p)
        return None
    json.dump(dict(map=s["map"], scale=s["scale"], window=bool(s["window"])), open(p, "w"), indent=1)
    return p


def read_sidecar(onnx_path: str) -> dict:
    p = sidecar_path(onnx_path)
    return spec_of(json.load(open(p)) if os.path.exists(p) else None)


def ckpt_spec(ckpt: str) -> dict:
    """The spec a checkpoint was trained with, from config.json beside it (runs/<rung>/<name>/); original path if none."""
    cfg = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "config.json")
    return spec_of(json.load(open(cfg))["env"].get("action") if os.path.exists(cfg) else None)


class NumpyMap:
    """CPU mirror (duel_stats.py, cpu_sim users): to_action(raw) -> a; target(...) -> ctrl. kp/kv/tau per joint of
    one robot, from the compiled model (nominal values)."""
    def __init__(self, s: dict, base_scale: float, kp=None, kv=None, tau=None):
        self.s = spec_of(s); self.scale = np.asarray(self.s["scale"], dtype=np.float64) if self.s["scale"] is not None else float(base_scale)
        self.kp, self.kv, self.tau = kp, kv, tau
        assert not self.s["window"] or kp is not None

    @staticmethod
    def gains(m, aid):
        """(kp, kv, tau) of actuators aid of a mujoco.MjModel (position actuators: biasprm = [0, -kp, -kv])."""
        return m.actuator_gainprm[aid, 0].copy(), -m.actuator_biasprm[aid, 2].copy(), m.actuator_forcerange[aid, 1].copy()

    def to_action(self, raw: np.ndarray) -> np.ndarray:
        return (np.tanh(raw) if self.s["map"] == "tanh" else np.clip(raw, -1, 1)).astype(np.float32)

    def target(self, base: np.ndarray, f: np.ndarray, q: np.ndarray, qd: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """base = default (+ style offset), f = the (filtered) action a."""
        t = base + self.scale * f
        if self.s["window"]: t = np.clip(t, q + (self.kv * qd - self.tau) / self.kp, q + (self.kv * qd + self.tau) / self.kp)
        return np.clip(t, lo, hi)
