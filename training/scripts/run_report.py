"""HTML report for a batch of training runs (AGENTS.md: runs over 30 min get a report with the top-3 metric charts,
deltas against the previous run and a three-tier summary). Charts are inline SVG drawn from the TensorBoard logs, so
the page is one self-contained file.
    uv run python scripts/run_report.py ../SCREENSHOTS/training-261019/spec.json
The spec (JSON) lists sections: {"title", "summary": {"toddler", "child", "adult"}, "runs": [{"label", "dir"}],
"baseline": run label to compute deltas against (optional), "metrics": [{"tag", "label", "scale"}] (three),
"evals": [{"label", "values": {column: text}}], "verdict"}; it is written next to index.html."""
import glob, html, json, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2"]


def series(run_dir, tag):
    files = glob.glob(os.path.join(ROOT, "runs", run_dir, "events.*"))
    if not files: return []
    ea = EventAccumulator(os.path.dirname(files[0]), size_guidance={"scalars": 0}); ea.Reload()
    if tag not in ea.Tags()["scalars"]: return []
    pts = [(s.step, s.value) for s in ea.Scalars(tag)]
    s0 = pts[0][0] if pts else 0
    return [(s - s0, v) for s, v in pts]                       # iterations from this run's start (resumed runs count on)


def smooth(pts, k=10):
    out = []
    for i in range(0, len(pts), k):
        c = pts[i:i + k]; out.append((sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c)))
    return out


def chart(runs, metric):
    W, H, P = 460, 220, 34
    data = [(r["label"], smooth([(s, v * metric.get("scale", 1.0)) for s, v in series(r["dir"], metric["tag"])])) for r in runs]
    data = [(l, d) for l, d in data if d]
    if not data: return f"<p class='muted'>{html.escape(metric['label'])}: no data</p>"
    xs = [p[0] for _, d in data for p in d]; ys = [p[1] for _, d in data for p in d]
    x0, x1, y0, y1 = 0, max(xs) or 1, min(ys), max(ys)
    if y1 - y0 < 1e-9: y1 = y0 + 1
    X = lambda x: P + (x - x0) / (x1 - x0) * (W - P - 8); Y = lambda y: H - 22 - (y - y0) / (y1 - y0) * (H - 34)
    lines = "".join(f"<polyline fill='none' stroke='{COLORS[i % len(COLORS)]}' stroke-width='2' points='{' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in d)}'/>"
                    for i, (_, d) in enumerate(data))
    legend = "".join(f"<span class='key'><i style='background:{COLORS[i % len(COLORS)]}'></i>{html.escape(l)}</span>" for i, (l, _) in enumerate(data))
    axes = (f"<line x1='{P}' y1='{H - 22}' x2='{W - 8}' y2='{H - 22}' class='ax'/><line x1='{P}' y1='12' x2='{P}' y2='{H - 22}' class='ax'/>"
            f"<text x='{P - 4}' y='16' class='tick' text-anchor='end'>{y1:.3g}</text><text x='{P - 4}' y='{H - 22}' class='tick' text-anchor='end'>{y0:.3g}</text>"
            f"<text x='{W - 8}' y='{H - 6}' class='tick' text-anchor='end'>{x1:.0f} it</text>")
    return (f"<figure><figcaption>{html.escape(metric['label'])}</figcaption><svg viewBox='0 0 {W} {H}' role='img' "
            f"aria-label='{html.escape(metric['label'])}'>{axes}{lines}</svg><div class='legend'>{legend}</div></figure>")


def deltas(sec):
    if not sec.get("baseline"): return ""
    base = next(r for r in sec["runs"] if r["label"] == sec["baseline"])
    rows = ""
    for m in sec["metrics"]:
        b = series(base["dir"], m["tag"]); bv = sum(v for _, v in b[-20:]) / max(len(b[-20:]), 1) * m.get("scale", 1.0) if b else None
        for r in sec["runs"]:
            if r is base: continue
            s = series(r["dir"], m["tag"])
            if not s or bv is None: continue
            v = sum(x for _, x in s[-20:]) / len(s[-20:]) * m.get("scale", 1.0)
            rows += f"<tr><td>{html.escape(m['label'])}</td><td>{html.escape(r['label'])}</td><td>{v:.4g}</td><td>{bv:.4g}</td><td>{v - bv:+.3g}</td></tr>"
    return ("<h3>Deltas against " + html.escape(sec["baseline"]) + " (last 20 iterations of training)</h3><table><tr><th>Metric</th><th>Run</th>"
            "<th>Value</th><th>Baseline</th><th>Delta</th></tr>" + rows + "</table>") if rows else ""


def section(sec):
    tiers = sec["summary"]
    cols = list(dict.fromkeys(c for e in sec.get("evals", []) for c in e["values"]))
    evals = ("<h3>Evaluation</h3><table><tr><th></th>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>" +
             "".join("<tr><td>" + html.escape(e["label"]) + "</td>" + "".join(f"<td>{html.escape(e['values'].get(c, ''))}</td>" for c in cols) + "</tr>"
                     for e in sec.get("evals", [])) + "</table>") if sec.get("evals") else ""
    return (f"<section><h2>{html.escape(sec['title'])}</h2>"
            f"<div class='tiers'><div><b>Toddler</b><p>{html.escape(tiers['toddler'])}</p></div><div><b>Child</b><p>{html.escape(tiers['child'])}</p></div>"
            f"<div><b>Adult</b><p>{html.escape(tiers['adult'])}</p></div></div>"
            f"<div class='charts'>{''.join(chart(sec['runs'], m) for m in sec['metrics'])}</div>{deltas(sec)}{evals}"
            f"<p class='verdict'>{html.escape(sec.get('verdict', ''))}</p></section>")


CSS = """:root{--bg:#f7f7f5;--ink:#1d1d1f;--dim:#5d5d63;--card:#fff;--line:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#151517;--ink:#ececef;--dim:#a0a0a8;--card:#1f1f23;--line:#333}}
body{margin:0;padding:24px 16px 48px;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:1000px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 10px}h3{font-size:15px;margin:18px 0 6px}.muted{color:var(--dim)}
.tiers{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.tiers div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tiers p{margin:4px 0 0;color:var(--dim)}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}figcaption{font-weight:600;font-size:14px}
svg{width:100%;height:auto}.ax{stroke:var(--dim);stroke-width:1}.tick{fill:var(--dim);font-size:10px}.legend{font-size:12px;color:var(--dim)}
.key{margin-right:10px;white-space:nowrap}.key i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}
.verdict{font-weight:600}"""


def main(spec_path):
    spec = json.load(open(spec_path))
    body = "".join(section(s) for s in spec["sections"])
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(spec['title'])}</title><style>{CSS}</style></head><body><main><h1>{html.escape(spec['title'])}</h1>"
            f"<p class='muted'>{html.escape(spec.get('subtitle', ''))}</p>{body}</main></body></html>")
    out = os.path.join(os.path.dirname(spec_path), "index.html"); open(out, "w", encoding="utf-8").write(page); print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
