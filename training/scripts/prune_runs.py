"""Prune abandoned / dummy runs, intermediate checkpoints and stale logs under training/runs (AGENTS.md housekeeping).
Dry run by default.

  uv run python scripts/prune_runs.py                 # list what would be deleted, and why
  uv run python scripts/prune_runs.py --apply         # delete it (--delete is the same)

A run directory is a folder holding TensorBoard event files, checkpoints (model_*.pt) or a config.json/manifest.json.
The whole directory is pruned when any of these holds:
  short      fewer than --min-steps environment steps logged: (last TensorBoard step + 1) x num_steps_per_env x
             num_envs from its config.json; without a config, the TensorBoard step count itself
  no-ckpt    no checkpoint at all (*.pt), and not touched for --idle-minutes
  abandoned  manifest.json says "running" or "failed" but nothing in it changed for --idle-minutes, and it has
             no model_final.pt
Inside a finished run (one with model_final.pt), the intermediate model_<iteration>.pt checkpoints are pruned.
A top-level runs/*.log is pruned when the run it is named after is finished or pruned, or after --log-days idle.
Never pruned: runs/league, runs/overnight, runs/audits, anything named in runs/league/champion.txt, every runs/...
path the queue scripts and the docs mention (scripts/**/*.sh, rl_optimization_log.md, STAGED_TRAINING.md, tasks.md,
CLAUDE.md), anything modified within --idle-minutes (a live run), and folders given with --keep."""
import argparse, glob, json, os, re, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
PROTECTED_ROOTS = ("league", "overnight", "audits")
REFERENCE_FILES = ["scripts/**/*.sh", "STAGED_TRAINING.md", "../rl_optimization_log.md", "../tasks.md", "../CLAUDE.md"]
CKPT = re.compile(r"model_(\d+)\.pt$")


def tb_last_step(d: str) -> int | None:
    """Largest step in the folder's TensorBoard event files (None when there are none or they cannot be read)."""
    files = glob.glob(os.path.join(d, "events.out.tfevents.*"))
    if not files: return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    last = -1
    for f in files:
        acc = EventAccumulator(f, size_guidance={"scalars": 0}); acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if tag.endswith("/time"): continue              # rsl_rl's "<x>/time" scalars are stepped by wall-clock seconds
            ev = acc.Scalars(tag)
            if ev: last = max(last, ev[-1].step)
    return last if last >= 0 else None


def env_steps(d: str, last: int | None) -> int | None:
    if last is None: return None
    try:
        c = json.load(open(os.path.join(d, "config.json")))
        return (last + 1) * int(c["train"]["num_steps_per_env"]) * int(c["args"]["num_envs"])
    except (OSError, KeyError, ValueError):
        return last + 1


def newest_mtime(d: str) -> float:
    return max((os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs), default=os.path.getmtime(d))


def size_of(p: str) -> int:
    if os.path.isfile(p): return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def run_dirs(runs: str):
    for r, dirs, files in os.walk(runs):
        if os.path.relpath(r, runs).split(os.sep)[0] in PROTECTED_ROOTS: dirs[:] = []; continue
        if any(f.startswith("events.out.tfevents.") or f.endswith(".pt") or f in ("config.json", "manifest.json") for f in files):
            dirs[:] = []; yield r


def referenced(runs: str) -> set[str]:
    """Absolute paths of every runs/... file or folder the scripts and docs mention (shell variables cut the match)."""
    refs = set()
    for pat in REFERENCE_FILES:
        for f in glob.glob(os.path.join(ROOT, pat), recursive=True):
            try: text = open(f, encoding="utf-8", errors="replace").read()
            except OSError: continue
            for m in re.findall(r"runs/[A-Za-z0-9_./-]+", text):
                refs.add(os.path.abspath(os.path.join(os.path.dirname(runs), m.rstrip("./"))))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs")); ap.add_argument("--min-steps", type=int, default=5000)
    ap.add_argument("--idle-minutes", type=float, default=30.0); ap.add_argument("--log-days", type=float, default=1.0)
    ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--apply", "--delete", dest="apply", action="store_true", help="really delete (default: dry run)")
    a = ap.parse_args()
    if not os.path.isdir(a.runs): print(f"no runs folder at {a.runs}"); return 0
    protected = set(os.path.abspath(k) for k in a.keep)
    refs = referenced(a.runs)
    champ = os.path.join(a.runs, "league", "champion.txt")
    champion = open(champ).read().strip() if os.path.exists(champ) else None
    now = time.time(); doomed = []                          # (path, reason)
    finished, pruned_runs = set(), set()
    for d in run_dirs(a.runs):
        idle = (now - newest_mtime(d)) / 60; ad = os.path.abspath(d)
        if os.path.exists(os.path.join(d, "model_final.pt")): finished.add(os.path.basename(d))
        if ad in protected or (champion and os.path.basename(d) == champion) or idle < a.idle_minutes:
            continue
        ckpts = glob.glob(os.path.join(d, "*.pt")); steps = env_steps(d, tb_last_step(d)); why = []
        if steps is not None and steps < a.min_steps: why.append(f"short ({steps} env steps)")
        if not ckpts: why.append("no-ckpt")
        try: status = json.load(open(os.path.join(d, "manifest.json"))).get("status")
        except (OSError, ValueError): status = None
        if status in ("running", "failed") and not os.path.exists(os.path.join(d, "model_final.pt")): why.append(f"abandoned (manifest: {status})")
        mentioned = any(r == ad or r.startswith(ad + os.sep) for r in refs)
        if why and not mentioned:
            doomed.append((d, ", ".join(why) + f", idle {idle:.0f} min")); pruned_runs.add(os.path.basename(d)); continue
        if os.path.exists(os.path.join(d, "model_final.pt")):
            for c in ckpts:
                if CKPT.search(c) and os.path.abspath(c) not in refs: doomed.append((c, "intermediate checkpoint"))
    for f in glob.glob(os.path.join(a.runs, "*.log")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem == "tensorboard" or os.path.abspath(f) in refs: continue
        days = (now - os.path.getmtime(f)) / 86400
        if stem in finished or stem in pruned_runs: doomed.append((f, f"log of {'a finished' if stem in finished else 'a pruned'} run"))
        elif days >= a.log_days: doomed.append((f, f"stale log ({days:.1f} days idle)"))
    total = files = 0; by_reason = {}
    for p, why in doomed:
        s = size_of(p); n = sum(len(fs) for _, _, fs in os.walk(p)) if os.path.isdir(p) else 1
        total += s; files += n; k = why.split(" (")[0].split(",")[0]; by_reason[k] = by_reason.get(k, 0) + s
        print(f"{'delete' if a.apply else 'would delete'} {os.path.relpath(p, a.runs)}: {why}, {s / 1e6:.1f} MB")
    for p, _ in doomed if a.apply else []:
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for k, s in sorted(by_reason.items(), key=lambda kv: -kv[1]): print(f"  {k}: {s / 1e6:.1f} MB")
    print(f"{files} file(s), {total / 1e6:.1f} MB {'deleted' if a.apply else 'would be deleted (dry run; pass --apply)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
