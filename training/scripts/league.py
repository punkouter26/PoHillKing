"""League training for the duel rungs (R4 mirror / R5). Generation k is warm-started from generation k-1 and
trained against a pool of the previous (up to 5) generations, each driving robot b_ with frozen weights.
After training, generation k is evaluated against that same pool.

  uv run python scripts/league.py --first 3 --last 6 [--iters 300] [--num-envs 3072]

Files: runs/league/gen<k>.pt (checkpoints), runs/league/results.jsonl (one line per generation).
If runs/r4league/gen<k>/model_final.pt already exists (or is being written by a running job) it is waited for
and reused instead of retraining."""
import argparse, glob, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LG = os.path.join(ROOT, "runs", "league")


def pool_for(k):
    return [os.path.join(LG, f"gen{i}.pt") for i in range(max(1, k - 5), k)]


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


ap = argparse.ArgumentParser()
ap.add_argument("--first", type=int, required=True); ap.add_argument("--last", type=int, required=True)
ap.add_argument("--iters", type=int, default=300); ap.add_argument("--num-envs", type=int, default=3072)
ap.add_argument("--eval-seconds", type=float, default=25.0)
a = ap.parse_args()
os.makedirs(LG, exist_ok=True)
for k in range(a.first, a.last + 1):
    out = os.path.join(LG, f"gen{k}.pt"); final = os.path.join(ROOT, "runs", "r4league", f"gen{k}", "model_final.pt")
    pool = pool_for(k)
    if not os.path.exists(out):
        running = os.path.isdir(os.path.dirname(final)) and not os.path.exists(final)
        if not os.path.isdir(os.path.dirname(final)):
            rc = run([PY, "-u", "scripts/train.py", "--rung", "r4league", "--num-envs", str(a.num_envs), "--iters", str(a.iters), "--name", f"gen{k}",
                      "--warm", os.path.join(LG, f"gen{k - 1}.pt"), "--opponent", *pool], os.path.join(ROOT, "runs", f"league_gen{k}.log"))
            if rc != 0: print(f"gen{k}: training failed (exit {rc})", flush=True); sys.exit(1)
        elif running:
            print(f"gen{k}: waiting for the running job", flush=True)
            while not os.path.exists(final): time.sleep(20)
            time.sleep(15)                                   # let the trainer finish its ONNX export and release the GPU
        shutil.copyfile(final, out)
    elog = os.path.join(ROOT, "runs", f"league_eval_gen{k}.log")
    if os.path.exists(elog): os.remove(elog)
    run([PY, "scripts/eval.py", "--rung", "r4league", "--ckpt", out, "--opponent", *pool, "--seeds", "3", "--num-envs", "256", "--seconds", str(a.eval_seconds)], elog)
    rows = [json.loads(l) for l in open(elog, encoding="utf-8") if l.startswith("{")]
    if not rows: print(f"gen{k}: evaluation produced no rows, see {elog}", flush=True); sys.exit(1)
    avg = lambda key: round(sum(r[key] for r in rows) / len(rows), 4)
    w, l, u = avg("attacker_ejects_defender"), avg("attacker_lost"), avg("undecided")
    res = dict(gen=k, pool=[os.path.basename(p) for p in pool], win=w, loss=l, tie=u, win_rate_of_decided=round(w / max(w + l, 1e-9), 4),
               median_time_s=avg("median_time_s"), passes_r5_bar=bool(w / max(w + l, 1e-9) > 0.55 and u < 0.15 and avg("median_time_s") < 20))
    with open(os.path.join(LG, "results.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)
