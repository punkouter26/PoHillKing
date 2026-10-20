"""Per-joint action scale from a drive audit (scripts/drive_audit.py --ckpt ... --json): the scale that makes +-1 cover
where the audited brain's raw outputs actually go, as the env cfg "action" scale (koth/action_map.py).

  uv run python scripts/action_scale.py runs/audits/drive_careful1.json --out runs/audits/scale_careful1.json

  scale_j = clip(base * p95(|raw a_j|), lo, reach_j), the same for a joint's left and right twin (max of the two p95s),
  reach_j = the farthest the joint's control range lets a target go from the default pose (a larger scale only adds clamp).
With base 0.5 (joint_map's action scale) a joint whose p95 is 1 keeps its scale; the champion's ankles, shoulders, elbows and
wrists, whose outputs sat on the clamp 25-65 % of the time, get more reach, the hips and waist finer steps."""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audit"); ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.25); ap.add_argument("--quantile-key", default="raw_abs_p95")
    a = ap.parse_args()
    rows = json.load(open(a.audit)); jm = json.load(open(os.path.join(ROOT, "assets", os.environ.get("KOTH_ROBOT", "g1"), "joint_map.json")))
    base = float(jm["action_scale"]); default = np.array(jm["default_pose"])
    lo = np.array([r["ctrlrange"][0] for r in rows]); hi = np.array([r["ctrlrange"][1] for r in rows])
    reach = np.maximum(np.abs(lo - default), np.abs(hi - default))
    p = {r["joint"]: r[a.quantile_key] for r in rows}
    twin = lambda j: j.replace("left_", "right_") if j.startswith("left_") else j.replace("right_", "left_")
    q = np.array([max(p[j], p.get(twin(j), p[j])) for j in jm["joints"]])
    scale = np.round(np.clip(base * q, a.lo, reach), 3)
    print(f"{'joint':>28} {'p95':>6} {'reach':>6} {'scale':>6}")
    for j, qq, rr, s in zip(jm["joints"], q, reach, scale): print(f"{j:>28} {qq:6.2f} {rr:6.2f} {s:6.3f}{'  (reach)' if s >= rr - 1e-3 else ''}")
    json.dump(dict(scale=scale.tolist(), joints=jm["joints"], source=os.path.relpath(a.audit, ROOT), rule=f"clip({base} x {a.quantile_key}, {a.lo}, reach), twins equal"),
              open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
