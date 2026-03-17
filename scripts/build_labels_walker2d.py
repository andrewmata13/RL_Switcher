"""
Build the adversarial-detection labeling dataset for Walker2D.

Clean obs → y=0, opt_attack(obs) → y=1   (50% each by construction).

Usage:
    python3.8 scripts/build_labels_walker2d.py \
        --perf-path   Walker2D/Walker2D_PPO.model \
        --attack-path Walker2D/Walker2D_Attack_PPO.model \
        --dataset-out data/walker2d_critical_dataset.npz \
        --episodes 20 --subsample-every 5
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np

from walker2d_ags_rs_switcher.controllers import Walker2DPerfPolicy
from walker2d_ags_rs_switcher.labeling import Walker2DCriticalBurstLabeler


def collect_state_stats(perf: Walker2DPerfPolicy, episodes: int = 5) -> tuple:
    states = []
    for _ in range(episodes):
        obs = perf.start_episode()
        done = False
        while not done:
            states.append(obs.astype("float32"))
            obs, _, done, _ = perf.step(perf.predict(obs))
    states = np.stack(states)
    return states.mean(axis=0), states.std(axis=0) + 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-path",       type=str, required=True)
    parser.add_argument("--attack-path",     type=str, required=True)
    parser.add_argument("--dataset-out",     type=str,
                        default="data/walker2d_critical_dataset.npz")
    parser.add_argument("--episodes",        type=int, default=20)
    parser.add_argument("--subsample-every", type=int, default=5)
    parser.add_argument("--naive-policy",    action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dataset_out) or ".", exist_ok=True)

    perf = Walker2DPerfPolicy.load(args.perf_path, attack_path=args.attack_path,
                                   naive_policy=args.naive_policy)

    print("Collecting observation statistics from clean PPO rollouts...")
    mean, std = collect_state_stats(perf, episodes=5)
    print(f"  obs mean (first 4): {mean[:4].round(3)}")
    print(f"  obs std  (first 4): {std[:4].round(3)}")

    labeler = Walker2DCriticalBurstLabeler(perf, cfg=None,
                                           state_mean=mean, state_std=std)

    print(f"\nBuilding detection dataset from {args.episodes} episodes "
          f"(subsample every {args.subsample_every} steps)...")
    X, y = labeler.build_dataset(
        n_episodes=args.episodes,
        subsample_every=args.subsample_every,
    )

    np.savez(args.dataset_out, X=X, y=y, state_mean=mean, state_std=std)
    frac = float(y.mean())
    print(f"\nSaved {args.dataset_out}:  {len(X)} samples,  "
          f"adversarial fraction = {frac:.3f}  (expect ~0.500)")


if __name__ == "__main__":
    main()
