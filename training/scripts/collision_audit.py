"""Self-collision audit of every fighter body (AGENTS.md section 3).

For G1, Kim and Trump (fighter a_ of scene_koth_2p_train.xml, the scene the duels train in) it measures, with mj_geomDistance, every pair of the body's
collision geoms that are not on the same or a parent-child body, in four sets of poses:
  zero    every joint at 0 (the physics zero pose)
  rest    the training keyframe (the standing pose every episode starts from)
  tpose   the rest pose with both shoulders rolled out to horizontal (sign found by measuring the hand)
  swing   every joint swept through its range in 9 steps, one at a time, the others at rest
A pair "collides in the simulation" when contype/conaffinity let it and no <exclude> removes it. The audit reports:
  - overlaps among the pairs the simulation does collide (must be none in zero, rest and T-pose)
  - overlaps among all non-adjacent pairs, i.e. what would touch if every self-collision were turned on (the
    AGENTS.md rule), so the cost of turning it on is known before retraining
Writes SCREENSHOTS/collision-<YYMMDD>/collision_audit.json, renders of the collision primitives (overlapping
geoms red) and report.html. Exit code 1 when an enabled pair overlaps in zero, rest or T-pose.

    uv run python scripts/collision_audit.py [--robots g1 kim trump] [--no-render]
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS = ROOT / "training" / "assets"
TOL = 1e-4          # m: deeper than this counts as overlap (contact solver margin aside)


def body_tree(m):
    """Adjacency: same body or parent-child."""
    def adjacent(b1, b2):
        return b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1
    return adjacent


def excluded(m, b1, b2):
    sig = {int(s) for s in m.exclude_signature[: m.nexclude]}
    return ((b1 << 16) + b2) in sig or ((b2 << 16) + b1) in sig


def robot_pairs(m, prefix="a_"):
    """Every pair of one fighter's collision geoms on non-adjacent bodies, and whether the simulation collides it."""
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])
             and name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).startswith(prefix)]
    adjacent = body_tree(m)
    pairs = []
    for i, a in enumerate(geoms):
        for b in geoms[i + 1:]:
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if adjacent(ba, bb):
                continue
            on = bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])) and not excluded(m, ba, bb)
            pairs.append((a, b, on))
    return geoms, pairs


def name(m, kind, i):
    return mujoco.mj_id2name(m, kind, i) or f"#{i}"


def overlaps(m, d, pairs):
    """[(geom a, geom b, enabled, depth m)] for every pair that interpenetrates."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)          # fills geom_xpos/xmat for mj_geomDistance
    fromto = np.zeros(6)
    out = []
    for a, b, on in pairs:
        dist = mujoco.mj_geomDistance(m, d, a, b, 0.05, fromto)
        if dist < -TOL:
            out.append((a, b, on, -dist))
    return out


def tpose(m, rest, prefix="a_"):
    """Rest pose with both shoulder rolls set so the arm is horizontal and out to the side."""
    q = rest.copy(); d = mujoco.MjData(m)
    for side in ("left", "right"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{side}_shoulder_roll_joint")
        hand = next((mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_{n}") for n in ("wrist_yaw_link", "wrist_roll_link", "elbow_link")
                     if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_{n}") >= 0), -1)
        assert j >= 0 and hand >= 0, f"no {prefix}{side} shoulder roll joint or hand"
        adr = m.jnt_qposadr[j]; lo, hi = m.jnt_range[j]
        best, best_q = -1e9, q[adr]
        for target in (np.pi / 2, -np.pi / 2):
            qq = q.copy(); qq[adr] = np.clip(target, lo, hi) if m.jnt_limited[j] else target
            d.qpos[:] = qq; mujoco.mj_kinematics(m, d)
            score = d.xpos[hand][2] + abs(d.xpos[hand][1])      # high and out to the side
            if score > best:
                best, best_q = score, qq[adr]
        q[adr] = best_q
    return q


def audit(robot: str):
    path = ASSETS / robot / "scene_koth_2p_train.xml"
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    geoms, pairs = robot_pairs(m)
    rest = m.key_qpos[0].copy() if m.nkey else m.qpos0.copy()
    zero = rest.copy()          # every hinge of fighter a_ at 0, its free joint and the other fighter as in the keyframe
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("a_"):
            zero[m.jnt_qposadr[j]] = 0.0
    poses = {"zero": zero, "rest": rest, "tpose": tpose(m, rest)}
    res = {"robot": robot, "model": str(path.relative_to(ROOT)).replace("\\", "/"), "collision_geoms": len(geoms),
           "pairs_non_adjacent": len(pairs), "pairs_enabled": sum(on for _, _, on in pairs),
           "enabled_pairs": sorted({f"{name(m, mujoco.mjtObj.mjOBJ_GEOM, a)} / {name(m, mujoco.mjtObj.mjOBJ_GEOM, b)}" for a, b, on in pairs if on}),
           "poses": {}, "swing": {}}
    for pose, q in poses.items():
        d.qpos[:] = q; ov = overlaps(m, d, pairs)
        res["poses"][pose] = {
            "enabled_overlaps": [(name(m, mujoco.mjtObj.mjOBJ_GEOM, a), name(m, mujoco.mjtObj.mjOBJ_GEOM, b), round(dep * 1000, 1)) for a, b, on, dep in ov if on],
            "all_overlaps": [(name(m, mujoco.mjtObj.mjOBJ_GEOM, a), name(m, mujoco.mjtObj.mjOBJ_GEOM, b), round(dep * 1000, 1)) for a, b, on, dep in ov],
        }
    # swing arcs: one hinge at a time through its range
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j] or not name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("a_"):
            continue
        adr = m.jnt_qposadr[j]; lo, hi = m.jnt_range[j]
        worst_all, worst_on, hits_all, hits_on, pairs_seen = 0.0, 0.0, 0, 0, {}
        for a in np.linspace(lo, hi, 9):
            q = rest.copy(); q[adr] = a; d.qpos[:] = q
            ov = overlaps(m, d, pairs)
            if ov:
                hits_all += 1
            if any(on for *_, on, _ in ov):
                hits_on += 1
            for ga, gb, on, dep in ov:
                key = f"{name(m, mujoco.mjtObj.mjOBJ_GEOM, ga)} / {name(m, mujoco.mjtObj.mjOBJ_GEOM, gb)}"
                pairs_seen[key] = max(pairs_seen.get(key, 0.0), dep)
                worst_all = max(worst_all, dep)
                if on:
                    worst_on = max(worst_on, dep)
        res["swing"][name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = {
            "samples": 9, "with_any_overlap": hits_all, "with_enabled_overlap": hits_on,
            "worst_any_mm": round(worst_all * 1000, 1), "worst_enabled_mm": round(worst_on * 1000, 1),
            "pairs": {k: round(v * 1000, 1) for k, v in sorted(pairs_seen.items(), key=lambda kv: -kv[1])[:4]},
        }
    return m, poses, pairs, res


def render(m, q, pairs, file: pathlib.Path):
    """Collision primitives only (visual meshes hidden): red = overlap on a pair the simulation collides, amber = overlap
    on a pair it does not (would touch with every self-collision on), blue = clear. Front and side view side by side."""
    d = mujoco.MjData(m); d.qpos[:] = q
    ov = overlaps(m, d, pairs)
    real = {g for a, b, on, _ in ov if on for g in (a, b)}             # pairs the simulation collides: red
    would = {g for a, b, on, _ in ov if not on for g in (a, b)} - real  # would touch with every self-collision on: amber
    rgba = m.geom_rgba.copy()
    for g in range(m.ngeom):
        coll = m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g]) and name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).startswith("a_")
        m.geom_rgba[g] = ([0.9, 0.15, 0.1, 0.95] if g in real else [0.95, 0.65, 0.15, 0.9] if g in would else [0.45, 0.65, 0.95, 0.75]) if coll else [0, 0, 0, 0]
    m.geom_rgba[0] = [0, 0, 0, 0]
    try:
        r = mujoco.Renderer(m, 480, 360)
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
        frames = []
        for az in (180.0, 90.0):
            pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a_pelvis")
            cam = mujoco.MjvCamera(); cam.lookat[:] = d.xpos[pel] + np.array([0, 0, 0.1]); cam.distance = 3.6; cam.azimuth = az; cam.elevation = -8
            mujoco.mj_forward(m, d); r.update_scene(d, cam, opt); frames.append(r.render())
        r.close()
    finally:
        m.geom_rgba[:] = rgba
    img = np.concatenate(frames, axis=1)
    try:
        from PIL import Image
        Image.fromarray(img).save(file)
    except ImportError:
        import imageio.v3 as iio
        iio.imwrite(file, img)


def report_html(results, shots, out: pathlib.Path):
    def esc(x):
        return html.escape(str(x))
    rows = []
    for r in results:
        p = r["poses"]
        rows.append(f"<tr><td>{esc(r['robot'])}</td><td>{r['collision_geoms']}</td><td>{r['pairs_non_adjacent']}</td><td>{r['pairs_enabled']}</td>"
                    + "".join(f"<td class='{'bad' if p[k]['enabled_overlaps'] else 'good'}'>{len(p[k]['enabled_overlaps'])}</td><td>{len(p[k]['all_overlaps'])}</td>" for k in ("zero", "rest", "tpose"))
                    + "</tr>")
    sections = []
    for r in results:
        sw = r["swing"]
        joints = sorted(sw.items(), key=lambda kv: -kv[1]["worst_any_mm"])
        jrows = "".join(f"<tr><td><code>{esc(j)}</code></td><td>{v['with_enabled_overlap']}/{v['samples']}</td><td>{v['worst_enabled_mm']}</td>"
                        f"<td>{v['with_any_overlap']}/{v['samples']}</td><td>{v['worst_any_mm']}</td><td>{esc(', '.join(f'{k} {mm} mm' for k, mm in v['pairs'].items()))}</td></tr>"
                        for j, v in joints if v["with_any_overlap"])
        tp = r["poses"]["tpose"]["all_overlaps"] + r["poses"]["rest"]["all_overlaps"]
        figs = "".join(f"<figure><img src='{esc(s.name)}' alt='{esc(r['robot'])} {esc(pose)} collision primitives'><figcaption>{esc(r['robot'])}, {esc(pose)}: front and side; red = overlap the simulation collides, amber = would overlap with every self-collision on</figcaption></figure>"
                       for (rb, pose), s in shots.items() if rb == r["robot"])
        sections.append(f"""
  <h2>{esc(r['robot'])}</h2>
  <p>{r['collision_geoms']} collision primitives, {r['pairs_non_adjacent']} pairs on non-adjacent bodies, of which the simulation collides
  {r['pairs_enabled']}: {esc(', '.join(r['enabled_pairs'][:12]) or 'none')}{' ...' if len(r['enabled_pairs']) > 12 else ''}.</p>
  <p>Rest and T-pose overlaps among all non-adjacent pairs: {esc(', '.join(f'{a} / {b} {mm} mm' for a, b, mm in tp) or 'none')}.</p>
  <div class="pair">{figs}</div>
  <table><tr><th>Joint swept</th><th>Enabled pairs overlap</th><th>Worst (mm)</th><th>Any pair overlaps</th><th>Worst (mm)</th><th>Deepest pairs</th></tr>{jrows or '<tr><td colspan=6>no overlap in any swing</td></tr>'}</table>""")
    out.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-Collision Audit</title>
<style>
  :root {{ --bg: #f6f6f4; --ink: #1d1d1f; --dim: #5d5d63; --card: #fff; --line: #ddd; --bad: #c0392b; --good: #1e8449; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #151517; --ink: #ececef; --dim: #a0a0a8; --card: #1f1f23; --line: #333; --bad: #ff6b5b; --good: #4cd18a; }} }}
  :root[data-theme="dark"] {{ --bg: #151517; --ink: #ececef; --dim: #a0a0a8; --card: #1f1f23; --line: #333; --bad: #ff6b5b; --good: #4cd18a; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; }}
  main {{ max-width: 1000px; margin: 0 auto; }} h2 {{ margin-top: 36px; }} p.sub {{ color: var(--dim); }}
  table {{ border-collapse: collapse; font-size: 14px; margin-top: 8px; display: block; overflow-x: auto; }}
  td, th {{ border-bottom: 1px solid var(--line); padding: 4px 10px 4px 0; text-align: left; vertical-align: top; }}
  .bad {{ color: var(--bad); font-weight: 600; }} .good {{ color: var(--good); }}
  .pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }}
  figure {{ margin: 0; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 8px; }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 6px; }} figcaption {{ font-size: 13px; color: var(--dim); }}
</style></head><body><main>
  <h1>Self-collision audit</h1>
  <p class="sub">{dt.date.today():%y%m%d}. <code>training/scripts/collision_audit.py</code>, MuJoCo {mujoco.__version__}, <code>mj_geomDistance</code> between collision primitives
  on bodies that are not the same or parent and child; overlap = deeper than {TOL * 1000:.1f} mm. "Enabled" = the simulation collides the pair
  (contype / conaffinity, no exclude). "Any" = every non-adjacent pair, which is what AGENTS.md section 3 asks to collide.</p>
  <table><tr><th>Body</th><th>Primitives</th><th>Non-adjacent pairs</th><th>Enabled</th>
  <th>Zero: enabled</th><th>any</th><th>Rest: enabled</th><th>any</th><th>T-pose: enabled</th><th>any</th></tr>{''.join(rows)}</table>
  {''.join(sections)}
</main></body></html>
""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", nargs="+", default=["g1", "kim", "trump"])
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "SCREENSHOTS" / f"collision-{dt.date.today():%y%m%d}"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results, shots, fail = [], {}, False
    for robot in a.robots:
        m, poses, pairs, res = audit(robot)
        results.append(res)
        for pose in ("zero", "rest", "tpose"):
            e = res["poses"][pose]["enabled_overlaps"]
            print(f"{robot:6s} {pose:6s} enabled overlaps {len(e)}  any {len(res['poses'][pose]['all_overlaps'])}" + (f"  {e}" if e else ""))
            fail |= bool(e)
        swing_on = sum(v["with_enabled_overlap"] > 0 for v in res["swing"].values()); swing_any = sum(v["with_any_overlap"] > 0 for v in res["swing"].values())
        print(f"{robot:6s} swing: {swing_on} of {len(res['swing'])} joints overlap on an enabled pair, {swing_any} on any pair")
        if not a.no_render:
            for pose in ("rest", "tpose"):
                f = out / f"{robot}_{pose}.png"
                try:
                    render(m, poses[pose], pairs, f); shots[(robot, pose)] = f
                except Exception as ex:  # no OpenGL context: the numbers still stand
                    print(f"render skipped ({ex})")
    (out / "collision_audit.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    report_html(results, shots, out / "report.html")
    print("wrote", out / "report.html")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
