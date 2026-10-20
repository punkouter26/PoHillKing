"""Success-gated curriculum: stages advance when the learner meets a bar, not after a fixed number of iterations.

Every curriculum so far ran on the clock (kicks 25 % -> 100 % over 300 iterations, the get-up pull fading over 1000,
the style ramp over 300). getup2 shows the failure mode: the help kept fading on schedule while the brain was stuck
sitting. Here a stage changes only when the metric over the last `window` iterations passes `bar` (and the stage has
lasted `min_iters`); a stage whose metric falls below `demote` steps back one stage, so a stage that is too hard is not
held until the policy settles below where it started (the cross-body league runs, log 2026-10-19).

Metric (from the env's outcome counters, learner rows, rounds that ended during the window):
  win_share   won / (won + lost + drawn)         duels and free-for-all
  survival    1 - fallen rows / rows             one robot (r0 / r1 style rungs)
  standing    the getup rung's standing share

A stage is a dict of overrides applied to the live env:
  pool        frozen-opponent draw weights (one per --opponent entry)
  push        disturbance scale (kicks and boxes, multiplies the rung's speeds)
  assist      get-up pull, x body weight
  reward      {term: weight}
Presets: ladder(k) for k opponents ordered weakest to strongest: stage i draws mostly opponent i, the last stage draws
all of them evenly (the full league)."""
from __future__ import annotations
import collections, json
import torch


def ladder(k: int, bar: float = 0.5, demote: float = 0.3) -> dict:
    stages = []
    for i in range(k):
        w = [0.0] * k; w[i] = 0.6
        for j in range(i): w[j] += 0.4 / max(i, 1)           # the weaker ones stay in the draw, so wins stay common
        if i == 0: w[0] = 1.0
        stages.append(dict(pool=w))
    stages.append(dict(pool=[1.0] * k))
    return dict(metric="win_share", bar=bar, demote=demote, window=20, min_iters=40, stages=stages)


PRESETS = {"ladder": ladder}


class Curriculum:
    def __init__(self, env, spec: dict, log=print):
        self.env, self.spec, self.log = env, spec, log
        self.stage, self.since = 0, 0
        self.hist = collections.deque(maxlen=spec.get("window", 20))
        self.trace = []                                   # (iteration, stage, metric) at every change
        self.apply(0)

    def apply(self, i: int):
        st = self.spec["stages"][i]; env = self.env
        if "pool" in st:
            assert env.frozen is not None and len(st["pool"]) == len(env.frozen.nets), "pool weights need one value per --opponent"
            env.frozen.set_weights(st["pool"])
        if "push" in st: env.push_scale_override = float(st["push"])
        if "assist" in st: env.cfg["getup"]["assist"] = float(st["assist"]); env.cfg["getup"]["assist_iters"] = 1e12
        for k, v in st.get("reward", {}).items(): env.cfg["reward"][k] = float(v)
        self.stage, self.since = i, 0; self.hist.clear()

    def metric(self, counts) -> float | None:
        m = self.spec["metric"]; won, lost, drew, rows, fallen, standing, steps = counts
        if m == "win_share": n = won + lost + drew; return won / n if n >= 1 else None
        if m == "survival": return 1 - fallen / max(rows, 1)
        if m == "standing": return standing / max(steps, 1)
        raise ValueError(m)

    def after_iteration(self, it: int):
        counts = self.env.read_outcomes()
        self.hist.append(counts); self.since += 1
        tot = [sum(c[k] for c in self.hist) for k in range(len(counts))]
        val = self.metric(tot)
        self.env.curriculum_log = dict(stage=float(self.stage), metric=float(val) if val is not None else -1.0)
        full = len(self.hist) == self.hist.maxlen and self.since >= self.spec.get("min_iters", 40)
        if val is None or not full: return
        last = len(self.spec["stages"]) - 1
        if val >= self.spec["bar"] and self.stage < last:
            self.log(f"curriculum: iteration {it} {self.spec['metric']} {val:.3f} >= {self.spec['bar']}: stage {self.stage} -> {self.stage + 1}")
            self.trace.append((it, self.stage + 1, round(val, 4))); self.apply(self.stage + 1)
        elif val < self.spec.get("demote", -1.0) and self.stage > 0:
            self.log(f"curriculum: iteration {it} {self.spec['metric']} {val:.3f} < {self.spec['demote']}: stage {self.stage} -> {self.stage - 1}")
            self.trace.append((it, self.stage - 1, round(val, 4))); self.apply(self.stage - 1)

    def summary(self) -> dict:
        return dict(spec=self.spec, final_stage=self.stage, stages=len(self.spec["stages"]), changes=self.trace)


def from_arg(arg: str, n_opponents: int) -> dict:
    """--curriculum ladder | ladder:bar:demote | path/to/spec.json"""
    if arg.endswith(".json"): return json.load(open(arg))
    name, *rest = arg.split(":")
    kw = dict(zip(("bar", "demote"), map(float, rest)))
    return PRESETS[name](n_opponents, **kw)
