import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np
import torch
import gymnasium as gym

from cartpole_ags_rs_switcher.config import LabelConfig
from cartpole_ags_rs_switcher.controllers import PerfPolicy
from cartpole_ags_rs_switcher.labeling import CriticalBurstLabeler


def collect_state_stats(perf_model, episodes: int = 30):
    env = gym.make("CartPole-v1")
    states = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            states.append(np.array(obs, dtype=np.float32))
            act, _ = perf_model.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
    env.close()
    states = np.stack(states)
    return states.mean(axis=0), states.std(axis=0) + 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-path", type=str, required=True)
    parser.add_argument("--dataset-out", type=str, default="data/critical_dataset.npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--subsample-every", type=int, default=4)
    parser.add_argument("--epsilon-l2", type=float, default=0.5, help="L2 radius of PGD attack in normalized observation space.")
    parser.add_argument("--burst-k", type=int, default=15)
    parser.add_argument("--horizon-h", type=int, default=100)
    parser.add_argument("--reward-drop-threshold", type=float, default=3.0)
    parser.add_argument("--pgd-steps", type=int, default=3, help="PGD steps per observation during labeling.")
    parser.add_argument("--n-attack-starts", type=int, default=3, help="Number of burst start positions tried per state.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dataset_out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perf = PerfPolicy.load(args.perf_path, device=device)
    mean, std = collect_state_stats(perf)

    cfg = LabelConfig(
        epsilon_l2=args.epsilon_l2,
        burst_k=args.burst_k,
        horizon_h=args.horizon_h,
        reward_drop_threshold=args.reward_drop_threshold,
        pgd_steps=args.pgd_steps,
        n_attack_starts=args.n_attack_starts,
    )
    labeler = CriticalBurstLabeler("CartPole-v1", perf, cfg, mean, std)
    X, y = labeler.build_dataset(perf.model, n_episodes=args.episodes, subsample_every=args.subsample_every)
    np.savez(args.dataset_out, X=X, y=y, state_mean=mean, state_std=std)
    print(f"saved dataset to {args.dataset_out} with {len(X)} samples; critical fraction={float(y.mean()):.3f}")


if __name__ == "__main__":
    main()
