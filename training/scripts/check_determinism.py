"""Bitwise check that koth/env.py (and scripts/eval.py) behave exactly like a reference version, plus throughput.

  uv run python scripts/check_determinism.py                      # reference = HEAD (git show HEAD:training/...)
  uv run python scripts/check_determinism.py --rungs r1 r4league --num-envs 64 --steps 200 --eval --bench 256

mujoco_warp is not run-to-run deterministic (its atomics: the same code differs by ~1e-6 after one control step and
chaotically after contact), and Warp's deterministic-atomics mode overflows its scatter buffers on this model. So
the physics is taken out of the comparison with a sim tape: the reference run records, at every _step_sim, the
inputs the torch side handed to the simulator (qpos, qvel, ctrl, every randomized model array) and what the torch
side reads back (qpos, qvel, xpos, contacts, nacon). The new run does not simulate: at each _step_sim it asserts its
inputs are bitwise equal to the recorded ones and then writes the recorded outputs. Everything the refactor touched
(resets, disturbances, box pinning, contacts, rewards, observations, frozen opponents, RNG stream) is therefore
compared bit for bit; the physics itself is unchanged code (the captured graph is the same).

The reference sources (koth/env.py, koth/obs.py, scripts/eval.py) are copied into a temp dir and imported next to
the working-tree ones. Each rung runs N worlds for S control steps with the same seed, closed loop with a randomly
initialised actor; per step qpos, qvel, observations (policy + critic), rewards, dones, time-outs, separation and
every reward term are compared with torch.equal. r4league uses a random rsl_rl-format checkpoint (written to the temp
dir) for the learner and as the frozen pool [ckpt, ckpt:stand, ckpt:shove] (stochastic, as in training): shared
module, stander, shove and opponent-noise paths. r0/r1 disturbances are made frequent (every 0.5-1.5 s) so the box
parking/pinning paths run. --eval compares eval.py's metric rows. --bench N times both versions (real physics).
Exit code 0 = identical."""
import argparse, importlib, json, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_show(rel: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:training/{rel}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def load_reference(tmp: str, env_src: str | None, obs_src: str | None, eval_src: str | None):
    """Import the reference env/obs/eval as koth_ref_env / koth_ref_obs / koth_ref_eval from tmp."""
    srcs = {"koth_ref_obs": open(obs_src).read() if obs_src else git_show("koth/obs.py"),
            "koth_ref_env": open(env_src).read() if env_src else git_show("koth/env.py"),
            "koth_ref_eval": open(eval_src).read() if eval_src else git_show("scripts/eval.py")}
    srcs["koth_ref_env"] = srcs["koth_ref_env"].replace("from koth.obs import", "from koth_ref_obs import")
    srcs["koth_ref_eval"] = srcs["koth_ref_eval"].replace("from koth.env import", "from koth_ref_env import")
    for name, s in srcs.items():
        with open(os.path.join(tmp, name + ".py"), "w", encoding="utf-8") as f: f.write(s)
    sys.path.insert(0, tmp)
    import koth.env
    env = importlib.import_module("koth_ref_env"); env.ASSETS = koth.env.ASSETS       # its path is relative to its own file
    return env, importlib.import_module("koth_ref_eval")


class SimTape:
    """Records (mode "record") or replays (mode "replay") every KothEnv._step_sim of one class."""
    def __init__(self, cls, mode: str, frames=None):
        self.cls, self.mode, self.frames, self.i, self.input_mismatch = cls, mode, frames if frames is not None else [], 0, []
        self.orig = cls._step_sim
        tape = self

        def _step_sim(env):
            ins = [env.qpos, env.qvel, env.ctrl] + [env.mp[k] for k in sorted(env.mp)]
            outs = [env.qpos, env.qvel, env.xpos, env.con_geom, env.con_world, env.con_dist, env.nacon]
            torch.cuda.synchronize()
            if tape.mode == "record":
                fin = [t.cpu().clone() for t in ins]; tape.orig(env)
                tape.frames.append((fin, [t.cpu().clone() for t in outs]))
            else:
                fin, fout = tape.frames[tape.i]
                for k, (a, b) in enumerate(zip(fin, ins)):
                    if not torch.equal(a, b.cpu()): tape.input_mismatch.append((tape.i, k))
                for dst, src in zip(outs, fout): dst.copy_(src.to(dst.device))
                torch.cuda.synchronize(); tape.i += 1
        cls._step_sim = _step_sim

    def remove(self):
        self.cls._step_sim = self.orig


def random_checkpoint(path: str, rung: str, out_scale: float = 1.0) -> None:
    """A randomly initialised rsl_rl checkpoint (OnPolicyRunner.save format) for the rung's observation width,
    with a non-trivial normalizer so (obs - mean) / (std + eps) is exercised. out_scale < 1 shrinks the actor's
    output layer (a near-zero action that mostly stands, so eval.py's tracking-error path runs too)."""
    from rsl_rl.runners import OnPolicyRunner
    from koth.env import KothEnv, default_cfg
    from scripts.train import train_cfg
    env = KothEnv(default_cfg("r4sp" if rung.startswith("r4") else rung), 8, seed=123)
    runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cuda")
    g = torch.Generator().manual_seed(7)
    for model in (runner.alg._raw_actor, runner.alg._raw_critic):
        n = model.obs_normalizer
        n._mean.copy_(torch.randn(n._mean.shape, generator=g) * 0.1); n._std.copy_(torch.rand(n._std.shape, generator=g) * 1.5 + 0.5)
    last = [m for m in runner.alg._raw_actor.mlp if isinstance(m, torch.nn.Linear)][-1]
    last.weight.data.mul_(out_scale); last.bias.data.mul_(out_scale)
    runner.save(path)
    del runner, env; torch.cuda.empty_cache()


def make_cfg(mod, rung: str, ckpt: str | None):
    cfg = mod.default_cfg(rung)
    if rung in ("r0", "r1"):
        cfg["push"].update(interval_s=[0.5, 1.5]); cfg["projectile"].update(interval_s=[0.5, 1.5])
    if rung.startswith("r4") and ckpt:
        cfg["frozen_opponent"] = [ckpt, ckpt + ":stand", ckpt + ":shove"]; cfg["frozen_stochastic"] = True
    return cfg


KEYS = ("qpos", "qvel", "policy", "critic", "rew", "done", "time_outs", "sep")


def rollout(mod, rung, ckpt, N, steps, seed=0):
    from koth.actor import load
    env = mod.KothEnv(make_cfg(mod, rung, ckpt), N, seed=seed)
    pol = load(ckpt, device="cuda")
    obs = env.get_observations()
    rec = {k: [] for k in KEYS}; terms = {}
    first = {"obs0": obs["policy"].cpu().clone(), "critic0": obs["critic"].cpu().clone()}
    for _ in range(steps):
        with torch.no_grad(): act = pol(obs["policy"])
        obs, rew, done, ex = env.step(act)
        for k, v in (("qpos", env.qpos), ("qvel", env.qvel), ("policy", obs["policy"]), ("critic", obs["critic"]), ("rew", rew),
                     ("done", done), ("time_outs", ex["time_outs"]), ("sep", ex.get("sep"))):
            if v is not None: rec[k].append(v.cpu().clone())
        for k, v in env.reward_terms.items(): terms.setdefault(k, []).append(v.cpu().clone())
    out = {k: torch.stack(v) for k, v in rec.items() if v}
    out.update(first); out["terms"] = {k: torch.stack(v) for k, v in terms.items()}
    del env; torch.cuda.empty_cache()
    return out


def compare(a, b, prefix=""):
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b: bad.append(f"{prefix}{k}: missing on one side"); continue
        if isinstance(a[k], dict): bad += compare(a[k], b[k], prefix + k + "."); continue
        if a[k].shape != b[k].shape: bad.append(f"{prefix}{k}: shape {tuple(a[k].shape)} vs {tuple(b[k].shape)}"); continue
        if not torch.equal(a[k], b[k]):
            diff = (a[k].double() - b[k].double()).abs(); diff[diff.isnan()] = 0
            bad.append(f"{prefix}{k}: differs, max abs {float(diff.max()):.3e}, {int((a[k] != b[k]).sum())} elements")
    return bad


def taped(ref_cls, new_cls, fn_ref, fn_new):
    """Run fn_ref with the reference class recording, then fn_new with the new class replaying the same tape."""
    rec = SimTape(ref_cls, "record")
    try: r_ref = fn_ref()
    finally: rec.remove()
    rep = SimTape(new_cls, "replay", rec.frames)
    try: r_new = fn_new()
    finally: rep.remove()
    problems = [f"sim inputs differ at step/array {m}" for m in rep.input_mismatch[:5]]
    if rep.i != len(rec.frames): problems.append(f"new run took {rep.i} sim steps, reference {len(rec.frames)}")
    return r_ref, r_new, problems, len(rec.frames)


def bench(mod, rung, ckpt, N, steps=100):
    from koth.actor import load
    env = mod.KothEnv(make_cfg(mod, rung, ckpt), N, seed=0); pol = load(ckpt, device="cuda")
    obs = env.get_observations()
    for _ in range(10):
        with torch.no_grad(): obs, *_ = env.step(pol(obs["policy"]))
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(steps):
        with torch.no_grad(): obs, *_ = env.step(pol(obs["policy"]))
    torch.cuda.synchronize(); el = time.perf_counter() - t
    rate = env.num_envs * steps / el
    del env; torch.cuda.empty_cache()
    return rate


def eval_rows(env_mod, eval_mod, rung, ckpt, N, steps):
    from scripts.eval import load_policy
    cfg = env_mod.default_cfg(rung)              # eval.py's settings (not make_cfg's frequent disturbances)
    if rung.startswith("r4"): cfg["frozen_opponent"] = [ckpt, ckpt + ":stand"]
    if rung in ("r0", "r1"):
        cfg["push"] = dict(interval_s=[5.0, 5.0], vel=[2.0, 2.0]); cfg["projectile"].update(speed=[6.0, 6.0])
    else:
        cfg["max_radius"] = 1.7; cfg["frozen_stochastic"] = False
    env = env_mod.KothEnv(cfg, N, seed=0); policy = load_policy(env, ckpt); rows = []
    for seed in range(2):
        torch.manual_seed(1000 + seed)
        rows.append(eval_mod.eval_flat(env, policy, rung, steps) if rung in ("r0", "r1") else eval_mod.eval_duel(env, policy, steps, league=True))
    del env; torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["r1", "r4league"]); ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--bench", type=int, default=0, help="also time N worlds (0 = skip)")
    ap.add_argument("--bench-steps", type=int, default=100)
    ap.add_argument("--eval", action="store_true", help="also compare eval.py metric rows (500 steps, 2 seeds)")
    ap.add_argument("--ref-env", default=None); ap.add_argument("--ref-obs", default=None); ap.add_argument("--ref-eval", default=None)
    ap.add_argument("--out", default=None, help="save the new rollouts (torch.save) to this file")
    a = ap.parse_args()
    import koth.env as new
    import scripts.eval as new_eval
    tmp = tempfile.mkdtemp(prefix="koth_det_")
    ref, ref_eval = load_reference(tmp, a.ref_env, a.ref_obs, a.ref_eval)
    ok = True; saved = {}
    # every reference rung must produce the same cfg, apart from keys the new code added
    for rung in ("r0", "r1", "r2", "r3", "r4", "r4att", "r4sp", "r4league", "r4shove", "r4probe"):
        c_ref, c_new = ref.default_cfg(rung), new.default_cfg(rung)
        def strip(n, r):
            return {k: (strip(v, r[k]) if isinstance(v, dict) and isinstance(r.get(k), dict) else v) for k, v in n.items() if k in r}
        same = strip(c_new, c_ref) == c_ref
        print(f"cfg {rung}: {'same' if same else 'DIFFERENT'} (added keys: {sorted(set(c_new) - set(c_ref))})"); ok &= same
    for rung in a.rungs:
        ckpt = os.path.join(tmp, f"random_{rung}.pt"); random_checkpoint(ckpt, rung)
        r_ref, r_new, problems, n_sim = taped(ref.KothEnv, new.KothEnv, lambda: rollout(ref, rung, ckpt, a.num_envs, a.steps),
                                              lambda: rollout(new, rung, ckpt, a.num_envs, a.steps))
        bad = problems + compare(r_ref, r_new)
        resets = int(r_new["done"].sum())
        print(f"{rung}: {a.num_envs} worlds x {a.steps} steps ({n_sim} taped sim calls), {resets} done rows, "
              f"{int(r_new['terms']['termination'].ne(0).sum())} terminations, terms {sorted(r_new['terms'])}")
        print(f"  reference vs new: {'BITWISE IDENTICAL (sim inputs, qpos, qvel, obs, critic, rewards, dones, time-outs, terms)' if not bad else bad}")
        ok &= not bad; saved[rung] = r_new
        if a.eval:
            q = os.path.join(tmp, f"quiet_{rung}.pt"); random_checkpoint(q, rung, out_scale=0.05)
            e_ref, e_new, problems, _ = taped(ref.KothEnv, new.KothEnv, lambda: eval_rows(ref, ref_eval, rung, q, a.num_envs, 500),
                                              lambda: eval_rows(new, new_eval, rung, q, a.num_envs, 500))
            same = e_ref == e_new and not problems
            print(f"  eval rows reference: {json.dumps(e_ref)}\n  eval rows new      : {json.dumps(e_new)}\n  eval rows {'IDENTICAL' if same else 'DIFFER ' + str(problems)}")
            ok &= same
        if a.bench:
            b_ref = bench(ref, rung, ckpt, a.bench, a.bench_steps); b_new = bench(new, rung, ckpt, a.bench, a.bench_steps)
            print(f"  throughput at {a.bench} worlds (real physics): reference {b_ref:,.0f} -> new {b_new:,.0f} policy steps/s ({(b_new / b_ref - 1) * 100:+.1f} %)")
    if a.out: torch.save(saved, a.out)
    shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT", "IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
