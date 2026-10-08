"""Time-boxed champion league for the all-round combat brain.

  uv run python scripts/allround_league.py --hours 3 [--first 3] [--iters 350]

Each generation k:
  1. warm-start from the current champion, train as robot a_ against a pool: the standing walker (5 entries),
     gen1, gen6, gen7 and the three most recent all-rounders;
  2. evaluate on the three combat bars: standing walker at the rim, the fixed benchmark pool gens 3-7, and the
     current champion;
  3. promote to champion only if it wins more than half of its decided rounds against the champion, still ejects
     the standing walker in at least 85 % of rounds, and does not go out by itself more often than the champion
     did (2-point tolerance).
Stops when less time remains than one generation needs. Results: runs/league/allround_results.jsonl,
champion pointer: runs/league/champion.txt."""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
LG = os.path.join(ROOT, "runs", "league"); P = lambda n: os.path.join(LG, n)


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def evaluate(rung, ckpt, opponents, seconds, tag):
    log = os.path.join(ROOT, "runs", f"allround_eval_{tag}.log")
    if os.path.exists(log): os.remove(log)
    run([PY, "scripts/eval.py", "--rung", rung, "--ckpt", ckpt, "--opponent", *opponents, "--seeds", "3", "--num-envs", "256", "--seconds", str(seconds)], log)
    rows = [json.loads(l) for l in open(log, encoding="utf-8") if l.startswith("{")]
    if not rows: raise RuntimeError(f"evaluation failed, see {log}")
    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    w, l, u = avg("attacker_ejects_defender"), avg("attacker_lost"), avg("undecided")
    return dict(win=round(w, 4), loss=round(l, 4), tie=round(u, 4), share=round(w / max(w + l, 1e-9), 4), median_s=round(avg("median_time_s"), 2))


ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=3.0); ap.add_argument("--first", type=int, default=3)
ap.add_argument("--iters", type=int, default=350); ap.add_argument("--num-envs", type=int, default=3072)
ap.add_argument("--champion", default="allround1"); ap.add_argument("--champion-self-loss", type=float, default=0.066)
a = ap.parse_args()
deadline = time.time() + a.hours * 3600
champ, champ_self_loss = a.champion, a.champion_self_loss
allrounders = [n for n in ("allround1", "allround2") if os.path.exists(P(n + ".pt"))]
gen_seconds = None; k = a.first
while True:
    need = gen_seconds * 1.1 if gen_seconds else a.iters * 5.5 + 600
    if time.time() + need > deadline:
        print(f"stopping: {int((deadline - time.time()) / 60)} min left, a generation needs about {int(need / 60)} min", flush=True); break
    t0 = time.time(); name = f"allround{k}"
    pool = [P("walker.pt") + ":stand"] * 5 + [P("gen1.pt"), P("gen6.pt"), P("gen7.pt")] + [P(n + ".pt") for n in allrounders[-3:]]
    final = os.path.join(ROOT, "runs", "r4league", name, "model_final.pt")
    rc = run([PY, "-u", "scripts/train.py", "--rung", "r4league", "--num-envs", str(a.num_envs), "--iters", str(a.iters), "--name", name,
              "--warm", P(champ + ".pt"), "--opponent", *pool, "--reward", "self_edge=-5.0"], os.path.join(ROOT, "runs", f"league_{name}.log"))
    if rc != 0 or not os.path.exists(final): print(f"{name}: training failed (exit {rc})", flush=True); break
    shutil.copyfile(final, P(name + ".pt")); time.sleep(10)
    stand = evaluate("r4att", P(name + ".pt"), [os.path.join(ROOT, "runs", "r1", "r1_b", "model_final.pt")], 15, f"{name}_stand")
    bench = evaluate("r4league", P(name + ".pt"), [P(f"gen{i}.pt") for i in (3, 4, 5, 6, 7)], 25, f"{name}_bench")
    vs = evaluate("r4league", P(name + ".pt"), [P(champ + ".pt")], 25, f"{name}_vs_champion")
    promote = vs["share"] > 0.5 and stand["win"] >= 0.85 and stand["loss"] <= champ_self_loss + 0.02
    res = dict(name=name, from_champion=champ, vs_standing=stand, vs_benchmark=bench, vs_champion=vs, promoted=bool(promote),
               minutes=round((time.time() - t0) / 60, 1))
    with open(P("allround_results.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)
    allrounders.append(name)
    if promote: champ, champ_self_loss = name, stand["loss"]
    open(P("champion.txt"), "w").write(champ + "\n")
    gen_seconds = time.time() - t0; k += 1
print("champion:", champ, flush=True)
