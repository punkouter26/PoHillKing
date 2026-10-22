"""Unattended queue: the humans' free-for-all brains (rung <body>_hffa, 121 inputs), the game's three-way match.

For each of Kim, Trump and Grandpa: learner a_ warm-started from the body's shipped fighter, against the other two
humans' shipped fighters (one per frozen slot, FrozenPool ":slotN"), on scene_koth_3p_<me>_<other>_<other>_train.xml
(python -m koth.build_mjcf mix ..., KOTH_ROBOT=g1). Then eval.py on the same three-way rung for the new brain and the
fighter it started from, and the new brain in a duel against that fighter (the roster gives a human its ffa brain for
every match, duels included). Exported as Models/<body>/ffa_policy.onnx (+ ffa_value.onnx) only when it beats the
fighter in the three-way by PROMOTE_MARGIN and its duel against the fighter is no worse than the fighter's mirror duel
by more than DUEL_SLACK (seat a_ carries a bias of its own, so the mirror is the baseline). Unity gate
afterwards: ParityBatch.RunHumanFfaStats(body) vs eval.py --rung <body>_hffa (see STAGED_TRAINING.md section 3).

  .venv/Scripts/python.exe scripts/queue_hffa.py [--iters 400] [--num-envs 1536] [--bodies kim trump grandpa]
Progress: runs/overnight/progress_hffa.txt; eval rows: runs/overnight/<body>_hffa1_eval.json; TensorBoard on :6006.
"""
import argparse, json, os, socket, subprocess, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY): PY = sys.executable
LOG_DIR = os.path.join(ROOT, "runs", "overnight"); os.makedirs(LOG_DIR, exist_ok=True)
PROGRESS = os.path.join(LOG_DIR, "progress_hffa.txt")
HUMANS = ("kim", "trump", "grandpa")
# the checkpoints behind the shipped Models/<body>/fighter_policy.onnx (matched numerically on 2026-10-21)
FIGHTER = {"kim": "runs/r4league/kim_fight1/model_final.pt", "trump": "runs/trump_r4bull/trump_fight1/model_final.pt",
           "grandpa": "runs/r4league/grandpa_fight2/model_final.pt"}
DUEL_RUNG = {"kim": "r4league", "trump": "trump_r4bull", "grandpa": "r4league"}
PROMOTE_MARGIN, DUEL_SLACK = 0.03, 0.05


def note(msg):
    line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f: f.write(line + "\n")


def ensure_tensorboard():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", 6006)) == 0: return
    tb = os.path.join(ROOT, ".venv", "Scripts", "tensorboard.exe")
    cmd = [tb, "--logdir", "runs", "--port", "6006"] if os.path.exists(tb) else [PY, "-m", "tensorboard.main", "--logdir", "runs", "--port", "6006"]
    subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    note("started TensorBoard on http://localhost:6006")


def run(args, robot, log_path=None, capture=False):
    env = dict(os.environ, PYTHONUNBUFFERED="1", KOTH_ROBOT=robot)
    if capture: return subprocess.run([PY] + args, cwd=ROOT, env=env, capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        return subprocess.run([PY] + args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


def evaluate(robot, args):
    p = run(["scripts/eval.py"] + args + ["--seeds", "3", "--num-envs", "256", "--leg-contact-rule", "unity"], robot, capture=True)
    rows = [json.loads(l) for l in (p.stdout or "").splitlines() if l.strip().startswith("{")]
    if not rows: note(f"eval failed for {robot} {args[:4]}: {(p.stderr or '')[-400:]}")
    return rows


def mean(rows, key): return sum(r.get(key, 0.0) for r in rows) / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=400); ap.add_argument("--num-envs", type=int, default=1536)
    ap.add_argument("--bodies", nargs="+", default=list(HUMANS)); a = ap.parse_args()
    note(f"=== queue_hffa: {a.bodies}, {a.iters} iterations, {a.num_envs} worlds ===")
    ensure_tensorboard()
    for me in a.bodies:
        others = [h for h in HUMANS if h != me]
        scene = os.path.join(ROOT, "assets", "g1", f"scene_koth_3p_{me}_{'_'.join(others)}_train.xml")
        if not os.path.exists(scene):
            run(["-m", "koth.build_mjcf", "mix", me] + others, "g1", os.path.join(LOG_DIR, f"build_mix_3p_{me}.log"))
        rung, name = f"{me}_hffa", f"{me}_hffa1"
        opp = [f"{FIGHTER[o]}:slot{i + 1}" for i, o in enumerate(others)]
        ckpt = f"runs/{rung}/{name}/model_final.pt"
        if not os.path.exists(os.path.join(ROOT, ckpt)):
            note(f"{name}: training ({rung}, warm from {FIGHTER[me]}, opponents {others})")
            p = run(["scripts/train.py", "--rung", rung, "--name", name, "--num-envs", str(a.num_envs), "--iters", str(a.iters),
                     "--warm", FIGHTER[me], "--opponent"] + opp + ["--frozen-deterministic", "--critic-warmup", "15", "--reward", "draw=-20"],
                    me, os.path.join(LOG_DIR, f"{name}.log"))
            if p.returncode != 0: note(f"{name}: training FAILED (rc {p.returncode}), see runs/overnight/{name}.log"); continue
        ev = {"ffa_new": evaluate(me, ["--rung", rung, "--ckpt", ckpt, "--opponent"] + opp + ["--seconds", "20"]),
              "ffa_fighter": evaluate(me, ["--rung", rung, "--ckpt", FIGHTER[me], "--opponent"] + opp + ["--seconds", "20"]),
              "duel_vs_fighter": evaluate(me, ["--rung", DUEL_RUNG[me], "--combat2", "--ckpt", ckpt, "--opponent", FIGHTER[me], "--seconds", "25"]),
              "duel_mirror": evaluate(me, ["--rung", DUEL_RUNG[me], "--ckpt", FIGHTER[me], "--opponent", FIGHTER[me], "--seconds", "25"])}
        with open(os.path.join(LOG_DIR, f"{name}_eval.json"), "w", encoding="utf-8") as f: json.dump(ev, f, indent=1)
        new, base = mean(ev["ffa_new"], "learner_last_one_in"), mean(ev["ffa_fighter"], "learner_last_one_in")
        def share(rows):
            w, l = mean(rows, "attacker_ejects_defender"), mean(rows, "attacker_lost"); return w / max(w + l, 1e-6)
        duel, mirror = share(ev["duel_vs_fighter"]), share(ev["duel_mirror"])
        note(f"{name}: three-way last one in {new:.3f} (fighter {base:.3f}); duel vs fighter decided share {duel:.3f} (mirror {mirror:.3f})")
        if new >= base + PROMOTE_MARGIN and duel >= mirror - DUEL_SLACK:
            p = run(["scripts/export_policy.py", "--ckpt", ckpt, "--name", "ffa", "--value"], me, os.path.join(LOG_DIR, f"{name}_export.log"))
            note(f"{name}: PROMOTED -> Assets/PoKingHill/Models/{me}/ffa_policy.onnx" if p.returncode == 0 else f"{name}: export FAILED")
        else:
            note(f"{name}: kept as a run, not exported (needs +{PROMOTE_MARGIN:.2f} three-way and a duel within {DUEL_SLACK:.2f} of the mirror)")
    note("=== queue_hffa done ===")


if __name__ == "__main__":
    main()
