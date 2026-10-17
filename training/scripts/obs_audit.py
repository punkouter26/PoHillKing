"""Observation audit: are the inputs Unity feeds a brain inside what the brain saw in training?

  uv run python scripts/obs_audit.py --ckpt runs/league/careful1.pt --unity assets/g1/unity_obs_koth_2p.jsonl
  uv run python scripts/obs_audit.py --ckpt runs/league/careful1.pt --tag koth_2p --rung r4league --warp-steps 300

Inputs:
  the checkpoint's observation normalizer (obs_normalizer mean / std, i.e. the training distribution);
  Unity observations: a JSONL file, one observation array per line (written by the Unity side to
      training/assets/<robot>/unity_obs_<tag>.jsonl; --tag picks that path);
  Warp observations: sampled here from KothEnv on the rung's eval settings, driven by the checkpoint's own
      deterministic actor (observation noise off by default, as in Unity; --noise keeps it on).
Per input it reports z = (x - mean) / (std + 1e-2) statistics for each source: mean z, std z, and the share of
samples with |z| > --z-max (outside what the network was fitted on), plus drift = |mean(Unity) - mean(Warp)| in
training standard deviations, and ks = the two-sample Kolmogorov-Smirnov distance between the Unity and Warp
samples (catches sign flips and scale errors that leave the mean alone). Rows are sorted by the worst of the three
relative to its bar; --top limits them. --json writes the full table. Exit code 1 when any input exceeds a bar."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from koth.obs import OBS_DIM, COMBAT_DIM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINTS = ["l_hip_p", "l_hip_r", "l_hip_y", "l_knee", "l_ank_p", "l_ank_r", "r_hip_p", "r_hip_r", "r_hip_y", "r_knee", "r_ank_p", "r_ank_r",
          "waist_y", "waist_r", "waist_p", "l_sh_p", "l_sh_r", "l_sh_y", "l_elbow", "l_wr_r", "l_wr_p", "l_wr_y",
          "r_sh_p", "r_sh_r", "r_sh_y", "r_elbow", "r_wr_r", "r_wr_p", "r_wr_y"]
COMBAT = ["opp_dx", "opp_dy", "opp_dvx", "opp_dvy", "centre_dx", "centre_dy", "my_radius", "opp_radius", "opp_upright"]


def obs_names(dim: int) -> list[str]:
    n = ([f"linvel_{c}" for c in "xyz"] + [f"angvel_{c}" for c in "xyz"] + [f"gravity_{c}" for c in "xyz"] + ["cmd_vx", "cmd_vy", "cmd_wz"]
         + [f"q_{j}" for j in JOINTS] + [f"qd_{j}" for j in JOINTS] + [f"last_{j}" for j in JOINTS] + ["phase_cos_l", "phase_sin_l", "phase_cos_r", "phase_sin_r"])
    # widths: 103 locomotion, +9 per combat block (112, 121), +29 fatigue state (132, 141, 150)
    extra = dim - OBS_DIM; fatigue = extra >= 29 and (extra - 29) % COMBAT_DIM == 0 and extra % COMBAT_DIM != 0
    blocks = (extra - (29 if fatigue else 0)) // COMBAT_DIM
    for k in range(1, max(blocks, 0) + 1): n += [f"{c}{'' if k == 1 else k}" for c in COMBAT]
    if fatigue: n += [f"fatigue_{j}" for j in JOINTS]
    return (n + [f"in{i}" for i in range(len(n), dim)])[:dim]


def zstats(x: np.ndarray, mean: np.ndarray, std: np.ndarray, z_max: float) -> dict:
    z = (x - mean) / (std + 1e-2)
    return dict(n=len(x), z_mean=z.mean(0), z_std=z.std(0), oor=(np.abs(z) > z_max).mean(0), raw_mean=x.mean(0))


def ks_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per column max |CDF_a - CDF_b| (two-sample Kolmogorov-Smirnov statistic)."""
    out = np.zeros(a.shape[1])
    for i in range(a.shape[1]):
        x, y = np.sort(a[:, i]), np.sort(b[:, i]); g = np.concatenate([x, y])
        out[i] = np.abs(np.searchsorted(x, g, side="right") / len(x) - np.searchsorted(y, g, side="right") / len(y)).max()
    return out


def warp_samples(ckpt, rung, opponent, steps, num_envs, noise) -> np.ndarray:
    from koth.env import KothEnv
    from koth.actor import load
    from scripts.eval import eval_cfg
    cfg = eval_cfg(rung, opponent=opponent, noise=noise, leg_contact_rule="unity")
    env = KothEnv(cfg, num_envs, seed=0); pol = load(ckpt, device="cuda")
    obs = env.reset(); out = []
    for _ in range(steps):
        with torch.no_grad(): obs, *_ = env.step(pol(obs["policy"][:, :pol.obs_dim]))
        out.append(obs["policy"][:, :pol.obs_dim].cpu())
    return torch.cat(out).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--unity", default=None, help="JSONL of Unity observations")
    ap.add_argument("--tag", default=None, help="read training/assets/<robot>/unity_obs_<tag>.jsonl")
    ap.add_argument("--rung", default=None, help="rung for the Warp samples (default from the input width: 103 r1, 112 r4league, 121 r5ffa)")
    ap.add_argument("--opponent", nargs="+", default=None, help="frozen opponents for the Warp samples (default: the checkpoint itself)")
    ap.add_argument("--warp-steps", type=int, default=300, help="0 = no Warp samples"); ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--noise", action="store_true", help="keep training observation noise in the Warp samples")
    ap.add_argument("--z-max", type=float, default=4.0); ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--drift-bar", type=float, default=0.5); ap.add_argument("--oor-bar", type=float, default=0.01)
    ap.add_argument("--ks-bar", type=float, default=0.25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"][0].double().numpy(); std = sd["obs_normalizer._std"][0].double().numpy(); D = len(mean)
    names = obs_names(D); src = {}
    path = a.unity or (os.path.join(ROOT, "assets", os.environ.get("KOTH_ROBOT", "g1"), f"unity_obs_{a.tag}.jsonl") if a.tag else None)
    if path:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        rows = [r["obs"] if isinstance(r, dict) else r for r in rows]
        u = np.array([r[:D] for r in rows if len(r) >= D], dtype=np.float64)
        if len(u) < len(rows): print(f"warning: {len(rows) - len(u)} Unity rows narrower than the brain's {D} inputs were skipped")
        if len(rows) and len(rows[0]) != D: print(f"warning: Unity rows have {len(rows[0])} values, the brain reads {D}")
        src["unity"] = zstats(u, mean, std, a.z_max); unity_x = u; print(f"unity: {len(u)} observations from {path}")
    if a.warp_steps:
        rung = a.rung or {OBS_DIM: "r1", OBS_DIM + COMBAT_DIM: "r4league", OBS_DIM + 2 * COMBAT_DIM: "r5ffa"}.get(D, "r1")
        opp = a.opponent or ([a.ckpt] if rung.startswith(("r4", "r5")) else None)
        w = warp_samples(a.ckpt, rung, opp, a.warp_steps, a.num_envs, a.noise)
        src["warp"] = zstats(w, mean, std, a.z_max); warp_x = w; print(f"warp: {len(w)} observations, rung {rung}, noise {'on' if a.noise else 'off'}")
    if not src: ap.error("give --unity/--tag and/or --warp-steps > 0")
    drift = np.abs(src["unity"]["raw_mean"] - src["warp"]["raw_mean"]) / (std + 1e-2) if len(src) == 2 else np.zeros(D)
    ks = ks_distance(unity_x, warp_x) if len(src) == 2 else np.zeros(D)
    worst_oor = np.max([s["oor"] for s in src.values()], 0)
    order = np.argsort(-np.maximum.reduce([drift / max(a.drift_bar, 1e-9), worst_oor / max(a.oor_bar, 1e-9), ks / max(a.ks_bar, 1e-9)]))
    cols = [k for k in ("unity", "warp") if k in src]
    print(f"{'input':>16} " + " ".join(f"{k + ' z mean':>12} {k + ' z std':>11} {k + ' |z|>' + str(a.z_max):>14}" for k in cols) + (f" {'drift':>7} {'ks':>5}" if len(cols) == 2 else ""))
    for i in order[:a.top]:
        print(f"{names[i]:>16} " + " ".join(f"{src[k]['z_mean'][i]:12.2f} {src[k]['z_std'][i]:11.2f} {src[k]['oor'][i]:14.1%}" for k in cols)
              + (f" {drift[i]:7.2f} {ks[i]:5.2f}" if len(cols) == 2 else ""))
    bad = [names[i] for i in range(D) if drift[i] > a.drift_bar or worst_oor[i] > a.oor_bar or ks[i] > a.ks_bar]
    print(f"{len(bad)} of {D} inputs over the bars (drift > {a.drift_bar} sd, > {a.oor_bar:.0%} beyond |z| {a.z_max}, ks > {a.ks_bar}): {bad[:20]}")
    if a.json:
        json.dump({"inputs": [{"i": i, "name": names[i], "train_mean": float(mean[i]), "train_std": float(std[i]), "drift_sd": float(drift[i]), "ks": float(ks[i]),
                               **{f"{k}_{m}": float(src[k][m][i]) for k in cols for m in ("z_mean", "z_std", "oor", "raw_mean")}} for i in range(D)],
                   "bad": bad}, open(a.json, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
