"""PPO training with rsl_rl 5 on the KothEnv. Usage:
  uv run python scripts/train.py --rung r0 --num-envs 4096 --iters 1500 [--resume runs/r0/model_500.pt]
TensorBoard logs under training/runs/<rung>/<timestamp>."""
import argparse, json, os, sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rsl_rl.runners import OnPolicyRunner
from koth.env import KothEnv, default_cfg


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
            else: raise ValueError(f"cannot warm-start {name}.{k}: {tuple(v.shape)} -> {tuple(new[k].shape)}")
        model.load_state_dict(new); print(f"warm start {name}: {padded} tensors padded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None)
    ap.add_argument("--opponent", default=None, nargs="+", help="checkpoint(s) of frozen policies that drive robot b_ (one drawn per world)")
    ap.add_argument("--warm", default=None, help="checkpoint to warm-start from (weights only, pads new inputs)")
    a = ap.parse_args()
    cfg = default_cfg(a.rung)
    if a.opponent: cfg["frozen_opponent"] = [(os.path.abspath(o[:-6]) + ":stand") if o.endswith(":stand") else os.path.abspath(o) for o in a.opponent]; cfg["frozen_stochastic"] = True
    env = KothEnv(cfg, a.num_envs, seed=a.seed)
    run = a.name or time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.rung, run)
    os.makedirs(log_dir, exist_ok=True)
    json.dump({"env": cfg, "train": train_cfg(), "args": vars(a)}, open(os.path.join(log_dir, "config.json"), "w"), indent=1)
    runner = OnPolicyRunner(env, train_cfg(), log_dir=log_dir, device="cuda")
    if a.resume:
        runner.load(a.resume)
    if a.warm:
        warm_start(runner, a.warm)
    runner.learn(a.iters, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    runner.export_policy_to_onnx(log_dir, "policy.onnx")
    print("saved", log_dir)


if __name__ == "__main__":
    main()
