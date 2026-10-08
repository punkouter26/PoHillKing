"""Build training_report.html (repo root) from TensorBoard logs and the measured results below.
  uv run python scripts/make_report.py
The chart series are re-read from training/runs each time. The ability grid numbers are entered by hand from
eval.py / Unity gate outputs; update GRID when new evaluations are run."""
import glob, json, os, time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run, tag):
    files = glob.glob(os.path.join(ROOT, "runs", run, "events.*"))
    if not files: return []
    ea = EventAccumulator(os.path.dirname(files[0]), size_guidance={"scalars": 0}); ea.Reload()
    return [(s.step, s.value) for s in ea.Scalars(tag)] if tag in ea.Tags()["scalars"] else []


def smooth(pts, n):
    return [[round(sum(p[0] for p in c) / len(c)), round(sum(p[1] for p in c) / len(c), 4)] for c in (pts[i:i + n] for i in range(0, len(pts), n))]


D = {}
ep = []
for run, lo, hi in (("r0/r0_c", 0, 400), ("r1/r1_a", 400, 1200), ("r1/r1_b", 1200, 2700)):
    ep += [(s, v * 0.02) for s, v in load(run, "Train/mean_episode_length") if lo <= s < hi]
D["stay_up"] = smooth(ep, 15)
lin, ang = [], []
for run, lo, hi in (("r1/r1_a", 400, 1200), ("r1/r1_b", 1200, 2700)):
    lin += [(s, v / 0.02 * 100) for s, v in load(run, "rew/tracking_lin_vel") if lo <= s < hi]
    ang += [(s, v / 0.015 * 100) for s, v in load(run, "rew/tracking_ang_vel") if lo <= s < hi]
D["track_lin"], D["track_ang"] = smooth(lin, 15), smooth(ang, 15)
for name, run in (("ko_back", "r4att/r4att_b"), ("ko_stand", "r4att/r4att_c")):   # win reward per step / bonus 30 * 50 steps/s * 60 s
    D[name] = smooth([(s, v / 30.0 * 50 * 60) for s, v in load(run, "rew/win")], 6)

C = lambda v, p, l, w="", t=None: dict(v=v, p=p, l=l, w=w, t=t)
NA = lambda w="": C("n/a", 0, "na", w)
GRID = [
    dict(name="Walker", note="Main brain. 2,700 practice rounds. Used for rungs 0, 1 and 3.", cells=[
        C("99.4%", 99, "strong", "of 2 m/s kicks and 6 m/s boxes survived"),
        C("0.87 of 1.0 m/s", 87, "strong", "average error 0.14 m/s, bar is 0.15"),
        C("0.69 of 0.8 rad/s", 86, "strong", "average error 0.10 rad/s, bar is 0.2"),
        C("100%", 100, "strong", "stays inside the plateau, measured at round 1,200"),
        C("about 30%", 30, "weak", "returns from the slope one time in three; no longer required"),
        C("100%", 100, "strong", "pairs meet within 6 s, nobody walks off"),
        C("0%", 2, "fail", "bumps into the opponent and stops"),
        C("0%", 2, "fail", "stalemate"),
        C("Yes", 100, "strong", "standing within 1 mm, walking within 4 cm over 2 m", "Verified"),
    ]),
    dict(name="Climber", note="Copy of the Walker with 200 practice rounds on the hill. Shelved: agents now start on the summit.", cells=[
        C("inherited", 0, "na", "not re-measured"), C("inherited", 0, "na", "not re-measured"), C("inherited", 0, "na", "not re-measured"),
        C("100%", 100, "strong", "of robots that start on the plateau stay there"),
        C("93 to 100%", 96, "strong", "of robots that start on the slope get back"),
        NA("single-robot agent"), NA("single-robot agent"), NA("single-robot agent"),
        C("Yes", 100, "strong", "within 3 cm after a 2 m climb", "Verified"),
    ]),
    dict(name="Attacker (generation 1)", note="Copy of the Walker that also sees the opponent and the rim. 500 practice rounds.", cells=[
        C("inherited", 0, "na", "not re-measured"), C("inherited", 0, "na", "not re-measured"), C("inherited", 0, "na", "not re-measured"),
        NA("its job is to leave the centre"), NA("shelved"),
        C("Yes", 100, "strong", "reaches the opponent in about 1 s"),
        C("98.7%", 99, "strong", "of 768 test rounds, typically in 2.0 s; loses 1.3%"),
        C("even match", 50, "okay", "against a copy of itself: 82% of rounds decided, wins split evenly; never beat a centre-holding Walker", "Getting there"),
        C("Yes", 95, "strong", "inputs and brain match exactly; over 60 Unity rounds the share decided and the winning times match training", "Verified"),
    ]),
    dict(name="Duelist (generation 6)", note="League-trained against generations 1 to 5.", cells=[
        C("inherited", 0, "na"), C("inherited", 0, "na"), C("inherited", 0, "na"), NA(), NA("shelved"),
        C("Yes", 100, "strong", "rounds last about 5 s"),
        C("4 to 7%", 6, "fail", "forgot how to handle an opponent that stands still"),
        C("59% / 95%", 77, "strong", "wins 59% of decided rounds against the five earlier generations; against itself 95% of rounds are decided"),
        C("Not yet", 0, "na", "generation 3 was the one checked"),
    ]),
    dict(name="Duelist (generation 7)", note="Latest. Trained against generations 1 to 6 plus the standing Walker.", cells=[
        C("inherited", 0, "na"), C("inherited", 0, "na"), C("inherited", 0, "na"), NA(), NA("shelved"),
        C("Yes", 100, "strong", "rounds last about 4 s"),
        C("7%", 7, "fail", "62% of those rounds end in a standoff"),
        C("80%", 80, "strong", "of decided rounds won against generations 2 to 6, 10% ties; against itself only half the rounds are decided"),
        C("Not yet", 0, "na", "exported to Unity, statistics not re-run"),
    ]),
    dict(name="Stand-only v1", retired=True, note="Retired. First attempt, taught standing with no walking.", cells=[
        C("47%", 47, "weak", "could not step to recover"), C("0%", 2, "fail", "never taught"), C("0%", 2, "fail", "never taught"),
        NA(), NA(), NA(), NA(), NA(), C("Yes", 100, "strong", "matched training to under a micron", "Verified"),
    ]),
    dict(name="Sparring pair", retired=True, note="Retired. Two identical learners fighting each other from the start.", cells=[
        C("inherited", 0, "na"), C("inherited", 0, "na"), C("inherited", 0, "na"), NA(), NA(),
        C("Yes", 100, "strong", "they meet chest to chest"),
        NA("not tested"), C("8 to 30%", 20, "weak", "of rounds decided; the rest were standoffs"), C("Not yet", 0, "na"),
    ]),
]

html = open(os.path.join(ROOT, "scripts", "report_template.html"), encoding="utf-8").read()
html = html.replace("__DATA__", json.dumps(D)).replace("__GRID__", json.dumps(GRID)).replace("__STAMP__", time.strftime("%d %B %Y, %H:%M"))
out = os.path.normpath(os.path.join(ROOT, "..", "training_report.html"))
open(out, "w", encoding="utf-8").write(html)
print("wrote", out, f"({len(html) // 1024} KB)", {k: len(v) for k, v in D.items()})
