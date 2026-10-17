"""Shared helpers for the league scripts (league.py, allround_league.py, judge.sh).

  run(cmd, log)                       run a command from training/, appending stdout+stderr to log; returns the exit code
  evaluate(rung, ckpt, opponents, seconds, log)
                                      scripts/eval.py (3 seeds x 256 duels) into a fresh log; returns its JSON rows
  summarize(rows)                     averages: win, loss, tie, share (of decided rounds), median_s

  python -m scripts.league_util --summarize "<tag>"   < eval.py JSON lines on stdin
      prints  <tag>: win W%  loss L%  tie T%  share of decided S%  median M s   (judge.sh)"""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(cmd, log) -> int:
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def rows_of(lines) -> list[dict]:
    return [json.loads(l) for l in lines if l.startswith("{")]


def evaluate(rung, ckpt, opponents, seconds, log, seeds=3, num_envs=256) -> list[dict]:
    """eval.py rows for ckpt vs the opponent pool; raises RuntimeError when eval.py printed none."""
    if os.path.exists(log): os.remove(log)
    run([PY, "scripts/eval.py", "--rung", rung, "--ckpt", ckpt, "--opponent", *opponents, "--seeds", str(seeds), "--num-envs", str(num_envs),
         "--seconds", str(seconds)], log)
    rows = rows_of(open(log, encoding="utf-8"))
    if not rows: raise RuntimeError(f"evaluation produced no rows, see {log}")
    return rows


def summarize(rows) -> dict:
    """Seed-averaged duel outcome (unrounded): win, loss, tie, share of decided rounds won, median winning time."""
    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    w, l, u = avg("attacker_ejects_defender"), avg("attacker_lost"), avg("undecided")
    return dict(win=w, loss=l, tie=u, share=w / max(w + l, 1e-9), median_s=avg("median_time_s"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--summarize", metavar="TAG", required=True); a = ap.parse_args()
    r = rows_of(sys.stdin); s = summarize(r); w, l = s["win"], s["loss"]
    print(f'{a.summarize}: win {w:.1%}  loss {l:.1%}  tie {1 - w - l:.1%}  share of decided {s["share"]:.1%}  median {s["median_s"]:.1f} s')
