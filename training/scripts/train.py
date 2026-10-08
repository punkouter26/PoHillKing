"""PPO training with rsl_rl 5 on the KothEnv. Usage:
  uv run python scripts/train.py --rung r0 --num-envs 4096 --iters 1500 [--resume runs/r0/model_500.pt]
TensorBoard logs under training/runs/<rung>/<timestamp>."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rsl_rl.runners import OnPolicyRunner
from koth.env import KothEnv, default_cfg


def train_cfg(num_steps_per_env=24, save_interval=100):
    return {
        "num_steps_per_env": num_steps_per_env, "save_interval": save_interval, "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "clip_param": 0.2, "gamma": 0.99,
                      "lam": 0.95, "value_loss_coef": 1.0, "entropy_coef": 0.005, "learning_rate": 1e-3, "max_grad_norm": 1.0,
                      "schedule": "adaptive", "desired_kl": 0.01},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="r0"); ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    cfg = default_cfg(a.rung)
    env = KothEnv(cfg, a.num_envs, seed=a.seed)
    run = a.name or time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.rung, run)
    os.makedirs(log_dir, exist_ok=True)
    json.dump({"env": cfg, "train": train_cfg(), "args": vars(a)}, open(os.path.join(log_dir, "config.json"), "w"), indent=1)
    runner = OnPolicyRunner(env, train_cfg(), log_dir=log_dir, device="cuda")
    if a.resume:
        runner.load(a.resume)
    runner.learn(a.iters, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    runner.export_policy_to_onnx(log_dir, "policy.onnx")
    print("saved", log_dir)


if __name__ == "__main__":
    main()
