"""Unattended overnight loop (default 10 h): the humans' one-brain-per-body recipe (rung <body>_hill, see queue_hill.py),
iterated in rounds until the time is up, then the best weights are saved and exported.

Round 1 warm-starts every body from its shipped fighter, against the other two shipped fighters. Every later round
continues each body from its best brain so far, against the other bodies' best brains (co-evolution): a run named
<body>_hill<round + 1>. A run becomes the body's new best when, on three-way rounds against the same opponents, it is
last one in at least MARGIN more often than the current best (round 1: the fighter, +0.03; later rounds +0.01), its duel
against the shipped fighter stays within DUEL_SLACK of the fighter's mirror duel, and it moves with at most JITTER_RATIO of
the fighter's joint-target change per step (queue_hill.py bars). A run starts only when the time left covers it.

Telemetry (runs/overnight/): progress_overnight.txt (one line per event), overnight_telemetry.jsonl (per run: reward
milestones, loss curve every 10 iterations, checkpoints, eval rows), overnight_summary.json (the best brain per body and the
round it came from); TensorBoard on :6006 has every curve. At the end each best brain is copied to
runs/league/<body>_hill_best.pt and exported over Models/<body>/fighter_policy.onnx (+ value, + fighter_action.json with the
low-pass) when it is not the shipped fighter. Unity afterwards: GameScene.RefreshBrains (the roster must pick up
fighter_action.json), then RunDuelStats / RunHumanFfaStats gates.

  .venv/Scripts/python.exe scripts/overnight_hill.py [--hours 10] [--iters 400] [--num-envs 1536]
"""
import argparse, glob, json, os, re, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue_hill as q

q.PROGRESS = os.path.join(q.LOG_DIR, "progress_overnight.txt")
TELEMETRY = os.path.join(q.LOG_DIR, "overnight_telemetry.jsonl")
SUMMARY = os.path.join(q.LOG_DIR, "overnight_summary.json")
MILESTONES = (0, 5, 10, 15, 20, 25, 30, 40)
ESTIMATE_MIN = 85            # one body's run: ~65 min training (400 it at 1536 worlds on the RTX 2060) + ~15 min of evals


def telemetry(log_path, run_dir):
    """Per-iteration values from a train.py log: reward milestones, a loss curve every 10 iterations, checkpoints."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log_path, encoding="utf-8", errors="ignore").read())
    rows = []
    for block in re.split(r"Learning iteration ", txt)[1:]:
        it = int(block.split("/", 1)[0]); r = {"it": it}
        for key, name in (("Mean reward", "reward"), ("Mean value loss", "value_loss"), ("Mean surrogate loss", "surrogate_loss"),
                          ("Mean entropy loss", "entropy"), ("Mean episode length", "ep_len")):
            m = re.search(re.escape(key) + r":\s*(-?[\d.]+)", block)
            if m: r[name] = float(m.group(1))
        rows.append(r)
    miles = {str(t): next((r["it"] for r in rows if r.get("reward", -1e9) >= t), None) for t in MILESTONES}
    return dict(iterations=len(rows), final=rows[-1] if rows else None, reward_milestones=miles,
                curve=[r for r in rows if r["it"] % 10 == 0], checkpoints=sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "model_*.pt"))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hours", type=float, default=10.0); ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--num-envs", type=int, default=1536); a = ap.parse_args()
    t0 = time.time(); deadline = t0 + a.hours * 3600
    left = lambda: (deadline - time.time()) / 60
    q.note(f"=== overnight_hill: {a.hours} h, {a.iters} iterations x {a.num_envs} worlds per run, bodies {list(q.HUMANS)} ===")
    q.ensure_tensorboard()
    best = {b: dict(ckpt=q.FIGHTER[b], round=0, three=None) for b in q.HUMANS}
    base = {}                                  # per body: the fighter's mirror duel share and three-way jitter (fixed references)
    for b in q.HUMANS:
        mirror = q.evaluate(b, ["--rung", q.DUEL_RUNG[b], "--ckpt", q.FIGHTER[b], "--opponent", q.FIGHTER[b], "--seconds", "25"])
        others = [h for h in q.HUMANS if h != b]
        three = q.evaluate(b, ["--rung", f"{b}_hill", "--ckpt", q.FIGHTER[b], "--opponent"] + [f"{q.FIGHTER[o]}:slot{i + 1}" for i, o in enumerate(others)] + ["--seconds", "20"])
        base[b] = dict(mirror=q.share(mirror), jitter=q.mean(three, "jitter_rad"), three=q.mean(three, "learner_last_one_in"))
        best[b]["three"] = base[b]["three"]
        q.note(f"baseline {b}: fighter three-way {base[b]['three']:.3f}, mirror duel {base[b]['mirror']:.3f}, jitter {base[b]['jitter']:.4f} rad/step")
    rnd = 0
    while True:
        rnd += 1
        for me in q.HUMANS:
            if left() < ESTIMATE_MIN:
                q.note(f"{left():.0f} min left: no time for another run"); break
            others = [h for h in q.HUMANS if h != me]
            opp = [f"{best[o]['ckpt']}:slot{i + 1}" for i, o in enumerate(others)]
            rung, name = f"{me}_hill", f"{me}_hill{rnd + 1}"
            run_dir = os.path.join(q.ROOT, "runs", rung, name); ckpt = f"runs/{rung}/{name}/model_final.pt"
            log = os.path.join(q.LOG_DIR, f"{name}.log")
            q.note(f"round {rnd} {name}: warm from {best[me]['ckpt']}, opponents {[best[o]['ckpt'] for o in others]}")
            p = q.run(["scripts/train.py", "--rung", rung, "--name", name, "--num-envs", str(a.num_envs), "--iters", str(a.iters),
                       "--warm", best[me]["ckpt"], "--opponent"] + opp + ["--frozen-deterministic", "--critic-warmup", "15", "--reward", "draw=-20"], me, log)
            tel = telemetry(log, run_dir) if os.path.exists(log) else {}
            if p.returncode != 0 or not os.path.exists(os.path.join(q.ROOT, ckpt)):
                q.note(f"{name}: training FAILED (rc {p.returncode})"); continue
            m = tel.get("reward_milestones", {})
            q.note(f"{name}: trained, final reward {tel['final'].get('reward') if tel.get('final') else '?'}, reward milestones (iteration) {m}")
            three_args = ["--rung", rung, "--opponent"] + opp + ["--seconds", "20"]
            ev = {"three_new": q.evaluate(me, three_args[:2] + ["--ckpt", ckpt] + three_args[2:]),
                  "three_best": q.evaluate(me, three_args[:2] + ["--ckpt", best[me]["ckpt"]] + three_args[2:]),
                  "duel_vs_fighter": q.evaluate(me, ["--rung", q.DUEL_RUNG[me], "--ckpt", ckpt, "--opponent", q.FIGHTER[me], "--seconds", "25"])}
            new, old = q.mean(ev["three_new"], "learner_last_one_in"), q.mean(ev["three_best"], "learner_last_one_in")
            duel, jit = q.share(ev["duel_vs_fighter"]), q.mean(ev["three_new"], "jitter_rad")
            margin = q.PROMOTE_MARGIN if best[me]["round"] == 0 else 0.01
            ok = new >= old + margin and duel >= base[me]["mirror"] - q.DUEL_SLACK and jit <= q.JITTER_RATIO * base[me]["jitter"]
            q.note(f"{name}: three-way {new:.3f} vs best {old:.3f} (needs +{margin:.2f}); duel vs fighter {duel:.3f} (mirror {base[me]['mirror']:.3f}); "
                   f"jitter {jit:.4f} (fighter {base[me]['jitter']:.4f}) -> {'NEW BEST' if ok else 'kept as a run'}")
            if ok:
                best[me] = dict(ckpt=ckpt, round=rnd, three=new)
                shutil.copy(os.path.join(q.ROOT, ckpt), os.path.join(q.ROOT, "runs", "league", f"{me}_hill_best.pt"))
            with open(TELEMETRY, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(run=name, body=me, round=rnd, accepted=ok, telemetry=tel, eval=ev, three_new=new, three_best=old, duel=duel, jitter=jit)) + "\n")
            json.dump(dict(best=best, baseline=base, hours=(time.time() - t0) / 3600), open(SUMMARY, "w"), indent=1)
        else:
            continue
        break
    # final weights: the best brain of every body that beat its fighter, exported for the game
    for b in q.HUMANS:
        if best[b]["round"] == 0: q.note(f"final {b}: no run beat the shipped fighter; Models/{b}/fighter_policy.onnx unchanged"); continue
        p = q.run(["scripts/export_policy.py", "--ckpt", best[b]["ckpt"], "--name", "fighter", "--value"], b, os.path.join(q.LOG_DIR, f"{b}_overnight_export.log"))
        q.note(f"final {b}: {best[b]['ckpt']} (round {best[b]['round']}, three-way {best[b]['three']:.3f}) -> Models/{b}/fighter_policy.onnx"
               if p.returncode == 0 else f"final {b}: export FAILED, see runs/overnight/{b}_overnight_export.log")
    json.dump(dict(best=best, baseline=base, hours=(time.time() - t0) / 3600, done=True), open(SUMMARY, "w"), indent=1)
    q.note(f"=== overnight_hill done after {(time.time() - t0) / 3600:.1f} h ===")


if __name__ == "__main__":
    main()
