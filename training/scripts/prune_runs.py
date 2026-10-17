"""Prune abandoned / dummy run directories under training/runs (AGENTS.md housekeeping). Dry run by default.

  uv run python scripts/prune_runs.py                 # list what would be deleted, and why
  uv run python scripts/prune_runs.py --delete        # delete it

A run directory is a folder holding TensorBoard event files, checkpoints (model_*.pt) or a config.json/manifest.json.
It is pruned when any of these holds:
  short      fewer than --min-steps environment steps logged: (last TensorBoard step + 1) x num_steps_per_env x
             num_envs from its config.json; without a config, the TensorBoard step count itself
  no-ckpt    no checkpoint at all (*.pt), and not touched for --idle-minutes
  abandoned  manifest.json says "running" or "failed" but nothing in it changed for --idle-minutes, and it has
             no model_final.pt
Never pruned: runs/league (copied league checkpoints, champion.txt), anything named in runs/league/champion.txt,
anything modified within --idle-minutes (a live run), and folders given with --keep."""
import argparse, glob, json, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def run_dirs(runs: str):
    for r, dirs, files in os.walk(runs):
        if os.path.relpath(r, runs).split(os.sep)[0] == "league": dirs[:] = []; continue
        if any(f.startswith("events.out.tfevents.") or f.endswith(".pt") or f in ("config.json", "manifest.json") for f in files):
            dirs[:] = []; yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs")); ap.add_argument("--min-steps", type=int, default=5000)
    ap.add_argument("--idle-minutes", type=float, default=30.0); ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--delete", action="store_true", help="really delete (default: dry run)")
    a = ap.parse_args()
    if not os.path.isdir(a.runs): print(f"no runs folder at {a.runs}"); return 0
    protected = set(os.path.abspath(k) for k in a.keep)
    champ = os.path.join(a.runs, "league", "champion.txt")
    champion = open(champ).read().strip() if os.path.exists(champ) else None
    now = time.time(); total = 0; doomed = []
    for d in run_dirs(a.runs):
        rel = os.path.relpath(d, a.runs); idle = (now - newest_mtime(d)) / 60
        if os.path.abspath(d) in protected or (champion and os.path.basename(d) == champion) or idle < a.idle_minutes:
            continue
        ckpts = glob.glob(os.path.join(d, "*.pt")); steps = env_steps(d, tb_last_step(d)); why = []
        if steps is not None and steps < a.min_steps: why.append(f"short ({steps} env steps)")
        if not ckpts: why.append("no-ckpt")
        try: status = json.load(open(os.path.join(d, "manifest.json"))).get("status")
        except (OSError, ValueError): status = None
        if status in ("running", "failed") and not os.path.exists(os.path.join(d, "model_final.pt")): why.append(f"abandoned (manifest: {status})")
        if why:
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs); total += size
            doomed.append(d); print(f"{'delete' if a.delete else 'would delete'} {rel}: {', '.join(why)}, {size / 1e6:.1f} MB, idle {idle:.0f} min")
    for d in doomed if a.delete else []:
        shutil.rmtree(d)
    print(f"{len(doomed)} run folder(s), {total / 1e6:.1f} MB {'deleted' if a.delete else 'would be deleted (dry run; pass --delete)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
