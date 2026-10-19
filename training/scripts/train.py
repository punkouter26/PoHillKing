"""PPO training with rsl_rl 5 on the KothEnv. Usage:
  uv run python scripts/train.py --rung r0 --num-envs 4096 --iters 1500 [--resume runs/r0/model_500.pt]
TensorBoard logs under training/runs/<rung>/<timestamp>.

Every run writes runs/<rung>/<name>/manifest.json (cfg, git commit, seed, args, graduation bars, start/end time) and
tracks the graduation gate (AGENTS.md): mean episode reward over the 100 most recent episodes (rsl_rl's own
buffer) and NaN/inf gradients in any update. --graduate-reward R exports the final policy to Unity (export_policy.py,
<name>_policy.onnx) only if that mean is >= R at the end and no update had a NaN gradient. --tensorboard starts
`tensorboard --logdir runs` (port 6006) unless one is already listening."""
import argparse, datetime, json, os, socket, statistics, subprocess, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rsl_rl.runners import OnPolicyRunner
from koth.env import KothEnv, default_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_cfg(num_steps_per_env=24, save_interval=100):
    return {
        "num_steps_per_env": num_steps_per_env, "save_interval": save_interval, "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar", "std_range": [0.05, 1.0]}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "clip_param": 0.2, "gamma": 0.99,
                      "lam": 0.95, "value_loss_coef": 1.0, "entropy_coef": 0.002, "learning_rate": 1e-3, "max_grad_norm": 1.0,
                      "schedule": "adaptive", "desired_kl": 0.01},
    }


def warm_start(runner, ckpt_path):
    """Load actor/critic from a checkpoint whose observation was narrower. Tensors that differ only in the
    last (input) dimension are copied into the leading columns; new actor input weights are zeroed so the
    policy starts out behaving exactly like the old one. Optimizer state and iteration count are not loaded."""
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    for name, model in (("actor", runner.alg._raw_actor), ("critic", runner.alg._raw_critic)):
        new = model.state_dict(); padded = 0
        for k, v in ck[f"{name}_state_dict"].items():
            if k not in new: continue
            if new[k].shape == v.shape: new[k] = v
            elif new[k].dim() == v.dim() and new[k].shape[:-1] == v.shape[:-1] and new[k].shape[-1] > v.shape[-1]:
                if k.endswith("weight"): new[k] = torch.zeros_like(new[k])
                new[k][..., :v.shape[-1]] = v; padded += 1
                # a second 9-value combat block (121 inputs) means what the first one means: start its normalizer
                # statistics from the first block's instead of mean 0 / std 1 frozen at the checkpoint's sample count
                w = v.shape[-1]
                if k.startswith("obs_normalizer.") and new[k].shape[-1] - w == 9 and w >= 112: new[k][..., w:w + 9] = v[..., w - 9:w]
            else: raise ValueError(f"cannot warm-start {name}.{k}: {tuple(v.shape)} -> {tuple(new[k].shape)}")
        model.load_state_dict(new); print(f"warm start {name}: {padded} tensors padded")


class Graduation:
    """Graduation bookkeeping on a runner: last-100-episode mean reward after every iteration, NaN/inf gradients in
    every PPO update (parameter hooks OR-ing into one device flag, read once per update). Changes no numbers."""
    def __init__(self, runner, target: float | None):
        self.target, self.updates, self.nan_updates = target, 0, 0
        self.mean_reward = None; self.best = None; self.first_pass_it = None; self.it = None
        self._flag = torch.zeros((), dtype=torch.bool, device=runner.device)
        for model in (runner.alg._raw_actor, runner.alg._raw_critic):
            for prm in model.parameters():
                if prm.requires_grad: prm.register_hook(self._hook)
        update, log = runner.alg.update, runner.logger.log

        def update_(*a, **k):
            self._flag.zero_(); out = update(*a, **k); self.updates += 1
            if bool(self._flag): self.nan_updates += 1; print(f"graduation: NaN/inf gradient in update {self.updates}")
            return out

        def log_(*a, **k):
            log(*a, **k); buf = runner.logger.rewbuffer; self.it = k.get("it")
            if len(buf) == buf.maxlen:
                self.mean_reward = statistics.mean(buf); self.best = max(self.best if self.best is not None else -1e30, self.mean_reward)
                if self.target is not None and self.first_pass_it is None and self.mean_reward >= self.target: self.first_pass_it = self.it
        runner.alg.update, runner.logger.log = update_, log_

    def _hook(self, g):
        self._flag.logical_or_(~torch.isfinite(g).all())

    @property
    def passed(self) -> bool:
        return self.target is not None and self.mean_reward is not None and self.mean_reward >= self.target and self.nan_updates == 0

    def summary(self) -> dict:
        return dict(target_mean_reward_100_episodes=self.target, final_mean_reward_100_episodes=self.mean_reward, best_mean_reward_100_episodes=self.best,
                    first_iteration_at_target=self.first_pass_it, updates=self.updates, updates_with_nan_gradients=self.nan_updates, graduated=self.passed)


def critic_warmup(runner, updates: int):
    """For the first `updates` PPO updates only the critic learns: the actor's parameters (and its action std) take no
    gradient. A policy warm-started from an exported brain has a trained actor but a random value function, and
    advantages from that critic would otherwise push the actor away from what it already does well."""
    actor = [p for p in runner.alg._raw_actor.parameters() if p.requires_grad]; update = runner.alg.update; done = [0]
    def update_(*a, **k):
        frozen = done[0] < updates; lr = runner.alg.learning_rate
        for p in actor: p.requires_grad_(not frozen)
        try: return update(*a, **k)
        finally:
            for p in actor: p.requires_grad_(True)
            if frozen:   # the adaptive schedule would raise the rate while KL is 0; keep it where it was
                runner.alg.learning_rate = lr
                for g in runner.alg.optimizer.param_groups: g["lr"] = lr
            done[0] += 1
            if done[0] == updates: print(f"critic warm-up over after {updates} updates: the actor learns from now on")
    runner.alg.update = update_


def git_commit() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return dict(commit=sha, training_dir_dirty=dirty)
    except OSError:
        return dict(commit=None)


def start_tensorboard(logdir: str, port: int = 6006):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0: print(f"tensorboard: already listening on http://localhost:{port}"); return None
    log = open(os.path.join(logdir, "tensorboard.log"), "a")
    proc = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
    print(f"tensorboard: http://localhost:{port} (pid {proc.pid}, keeps running after training)")
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None)
    ap.add_argument("--opponent", default=None, nargs="+", help="checkpoint(s) of frozen policies that drive robot b_ (one drawn per world)")
    ap.add_argument("--rim-brake", type=float, nargs=3, default=None, help="rim gain vmin for the goal command")
    ap.add_argument("--reward", nargs="*", default=[], help="reward weight overrides, e.g. self_edge=-5.0")
    ap.add_argument("--warm", default=None, help="checkpoint to warm-start from (weights only, pads new inputs)")
    ap.add_argument("--decimation", type=int, default=None, help="physics steps per policy step (10 = 50 Hz, 15 = 33 Hz, 20 = 25 Hz)")
    ap.add_argument("--leg-contact-rule", choices=["train", "unity"], default="train", help="unity: own foot-shin contact does not end an episode")
    ap.add_argument("--push-out-cap", type=float, default=None, help="at most this much push_out reward per episode")
    ap.add_argument("--fatigue", choices=["off", "on", "obs"], default="off", help="actuator fatigue model (obs: also append the 29 fatigue states)")
    ap.add_argument("--combat2", action="store_true", help="append the second-nearest-opponent combat block (121 inputs)")
    ap.add_argument("--style-pose", default=None, help="the learner fights with this styled action path (bull: Trump's charge), e.g. to continue a styled brain on another rung")
    ap.add_argument("--graduate-reward", type=float, default=None, help="export to Unity if the last-100-episode mean reward reaches this")
    ap.add_argument("--export-name", default=None, help="Unity policy name for a graduated export (default: the run name)")
    ap.add_argument("--export-dir", default=None, help="folder for a graduated export (default: the Unity Models folder)")
    ap.add_argument("--tensorboard", action="store_true", help="start tensorboard --logdir runs")
    ap.add_argument("--critic-warmup", type=int, default=0, help="freeze the actor (and its std) for this many updates so a fresh critic can fit first (warm starts from ONNX have no critic)")
    a = ap.parse_args()
    t_start = time.time()
    cfg = default_cfg(a.rung)
    if a.decimation: cfg["decimation"] = a.decimation
    if a.leg_contact_rule == "unity": cfg["terminate_on_leg_contact"] = False
    if a.push_out_cap is not None: cfg["push_out_cap"] = a.push_out_cap
    if a.fatigue != "off": cfg["fatigue"].update(enabled=True, obs=a.fatigue == "obs")
    if a.combat2: cfg["combat2"] = True
    if a.style_pose: cfg.update(shove=True, style_pose=a.style_pose)
    if a.opponent: cfg["frozen_opponent"] = list(a.opponent); cfg["frozen_stochastic"] = True
    if a.rim_brake: cfg["goal_cmd"]["rim_brake"] = tuple(a.rim_brake)
    for kv in a.reward:
        k, v = kv.split("="); assert k in cfg["reward"], k; cfg["reward"][k] = float(v)
    env = KothEnv(cfg, a.num_envs, seed=a.seed)
    run = a.name or time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(ROOT, "runs", a.rung, run)
    os.makedirs(log_dir, exist_ok=True)
    json.dump({"env": cfg, "train": train_cfg(), "args": vars(a)}, open(os.path.join(log_dir, "config.json"), "w"), indent=1)
    stamp = lambda t: datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")
    manifest = dict(rung=a.rung, name=run, robot=os.environ.get("KOTH_ROBOT", "g1"), git=git_commit(), seed=a.seed, args=vars(a),
                    env_cfg=cfg, train_cfg=train_cfg(), start_time=stamp(t_start), end_time=None, status="running")
    write_manifest = lambda: json.dump(manifest, open(os.path.join(log_dir, "manifest.json"), "w"), indent=1)
    write_manifest()
    if a.tensorboard: start_tensorboard(os.path.join(ROOT, "runs"))
    runner = OnPolicyRunner(env, train_cfg(), log_dir=log_dir, device="cuda")
    store = runner.logger._store_code_state            # rsl_rl snapshots `git diff` of the repo; binary/non-UTF-8 diffs crash it
    def store_safely():
        try: return store()
        except (UnicodeError, OSError) as e: print(f"warning: rsl_rl could not store the git diff ({type(e).__name__}); continuing"); return []
    runner.logger._store_code_state = store_safely
    if a.resume:
        runner.load(a.resume)
    if a.warm:
        warm_start(runner, a.warm)
    if a.critic_warmup > 0:
        critic_warmup(runner, a.critic_warmup)
    env.curriculum_offset_iters = runner.current_learning_iteration      # disturbance curriculum continues on --resume
    grad = Graduation(runner, a.graduate_reward)
    try:
        runner.learn(a.iters, init_at_random_ep_len=True)
    except BaseException:
        manifest.update(status="failed", end_time=stamp(time.time()), bars=grad.summary()); write_manifest(); raise
    runner.save(os.path.join(log_dir, "model_final.pt"))
    runner.export_policy_to_onnx(log_dir, "policy.onnx")
    manifest.update(status="finished", end_time=stamp(time.time()), minutes=round((time.time() - t_start) / 60, 1),
                    iterations=runner.current_learning_iteration + 1, bars=grad.summary(), unity_export=None)
    if a.graduate_reward is not None:
        print("graduation:", json.dumps(grad.summary()))
        if grad.passed:
            from scripts.export_policy import export
            from koth.build_mjcf import UNITY_MODELS
            path = os.path.join(a.export_dir or UNITY_MODELS, f"{a.export_name or run}_policy.onnx")
            export(os.path.join(log_dir, "model_final.pt"), path); manifest["unity_export"] = path
        else:
            print("graduation: not passed, no Unity export")
    write_manifest()
    print("saved", log_dir)


if __name__ == "__main__":
    main()
