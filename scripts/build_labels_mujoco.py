"""
Build the adversarial-detection labeling dataset for MuJoCo environments.

Collects clean and adversarially-perturbed observations from PPO rollouts.
Clean obs -> y=0 (non-critical, use PPO).
Adversarial obs (opt_attack applied) -> y=1 (critical, use ATLA).

Usage:
    python3.8 scripts/build_labels_mujoco.py --env hopper \
        --perf-path   Hopper/Hopper_PPO.model \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --dataset-out data/hopper_critical_dataset.npz \
        --episodes 20 --subsample-every 5
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.controllers import MuJoCoPerfPolicy
from rs_switcher_common.labeling import CriticalBurstLabeler, collect_state_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment name")
    parser.add_argument("--perf-path",   type=str, required=True)
    parser.add_argument("--attack-path", type=str, required=True)
    parser.add_argument("--dataset-out", type=str, default=None,
                        help="Output path (default: data/{env}_critical_dataset.npz)")
    parser.add_argument("--episodes",    type=int, default=20)
    parser.add_argument("--subsample-every", type=int, default=5)
    parser.add_argument("--naive-policy", action="store_true",
                        help="Load policy from --perf-path with its own ZFilter "
                             "instead of the hardened policy from --attack-path")
    args = parser.parse_args()

    config = ENV_REGISTRY[args.env]
    dataset_out = args.dataset_out or f"data/{args.env}_critical_dataset.npz"

    os.makedirs(os.path.dirname(dataset_out) or ".", exist_ok=True)

    perf = MuJoCoPerfPolicy.load(config, args.perf_path,
                                  attack_path=args.attack_path,
                                  naive_policy=args.naive_policy)

    print("Collecting observation statistics from clean PPO rollouts...")
    mean, std = collect_state_stats(perf, episodes=5)
    print(f"  obs mean (first 4): {mean[:4].round(3)}")
    print(f"  obs std  (first 4): {std[:4].round(3)}")

    labeler = CriticalBurstLabeler(perf, cfg=None, state_mean=mean, state_std=std)

    print(f"\nBuilding detection dataset from {args.episodes} episodes "
          f"(subsample every {args.subsample_every} steps)...")
    X, y = labeler.build_dataset(
        n_episodes=args.episodes,
        subsample_every=args.subsample_every,
    )

    np.savez(dataset_out, X=X, y=y, state_mean=mean, state_std=std)
    frac = float(y.mean())
    print(f"\nSaved {dataset_out}:  {len(X)} samples,  "
          f"adversarial fraction = {frac:.3f}  (expect ~0.500)")


if __name__ == "__main__":
    main()
