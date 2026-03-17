"""
Evaluate Walker2D controllers under the Zhang et al. adversarial attack.

Compares three controllers:
  always_perf      — native PPO only
  always_backup    — ATLA only
  anytime_switcher — RS-certified detection + ATLA recovery + permanent PPO commit

Walker2D specifics vs HalfCheetah:
  obs_dim=17, action_dim=6, eps=0.05 (L-inf)
  L-2 budget = eps * sqrt(obs_dim) = 0.05 * sqrt(17) ≈ 0.206

Usage:
    python3.8 scripts/evaluate_burst_attack_walker2d.py \
        --perf-path   Walker2D/Walker2D_PPO.model \
        --attack-path Walker2D/Walker2D_Attack_PPO.model \
        --backup-path Walker2D/Walker2D_ATLA.model \
        --switcher-path models/walker2d_switcher.pt \
        --dataset  data/walker2d_critical_dataset.npz \
        --sigma 0.1 --n-samples 10000 \
        --delta-budget-l2 0.15 \
        --episodes 30 --seed 0 \
        --burst-k 75 --t-candidate-max 100 \
        --recovery-k 100 --commit-timeout-k 5 \
        --device cuda \
        --output-json results/walker2d_seed0.json
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import torch

from walker2d_ags_rs_switcher.controllers import Walker2DPerfPolicy, Walker2DBackupPolicy
from walker2d_ags_rs_switcher.evaluation import (
    AlwaysPerfController, AlwaysBackupController,
    AnyTimeSwitcherController,
    evaluate_controller,
)
from walker2d_ags_rs_switcher.models import SwitcherMLP
from walker2d_ags_rs_switcher.rs import VanillaRSSwitcher


def run(controller, perf, backup, episodes, seed, attacked, burst_k, t_candidate_max):
    returns, logs = evaluate_controller(
        controller, perf, backup,
        n_episodes=episodes, seed=seed, attack=attacked, burst_k=burst_k,
        t_candidate_max=t_candidate_max,
    )
    allow_means = [l["allow_perf"] for l in logs]
    R_means     = [l["R_exec"]     for l in logs if not np.isnan(l["R_exec"])]
    n_fell      = sum(l["fell"] for l in logs)
    return {
        "mean_return":     float(np.mean(returns)),
        "std_return":      float(np.std(returns)),
        "mean_allow_perf": float(np.mean(allow_means)),
        "mean_R_exec":     float(np.nanmean(R_means)) if R_means else float("nan"),
        "n_fell":          n_fell,
        "fall_rate":       n_fell / len(logs),
        "returns":         [float(r) for r in returns],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-path",        type=str, required=True)
    parser.add_argument("--attack-path",      type=str, required=True)
    parser.add_argument("--backup-path",      type=str, required=True)
    parser.add_argument("--switcher-path",    type=str, required=True)
    parser.add_argument("--dataset",          type=str, required=True)
    parser.add_argument("--sigma",            type=float, default=0.1)
    parser.add_argument("--n-samples",        type=int,   default=10000)
    parser.add_argument("--delta-budget-l2",  type=float, default=0.15)
    parser.add_argument("--episodes",         type=int,   default=10)
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--burst-k",          type=int,   default=75)
    parser.add_argument("--detection-k",      type=int,   default=2)
    parser.add_argument("--recovery-k",       type=int,   default=100)
    parser.add_argument("--commit-timeout-k", type=int,   default=5)
    parser.add_argument("--t-candidate-max",  type=int,   default=100)
    parser.add_argument("--device",           type=str,   default=None)
    parser.add_argument("--monitoring-delta",  type=float, default=None,
                        help="Phase-1 detection threshold (default: same as delta-budget-l2). "
                             "Set to 0.0 to use pred-only detection.")
    parser.add_argument("--naive-policy",     action="store_true")
    parser.add_argument("--attack-eps",       type=float, default=None)
    parser.add_argument("--output-json",      type=str,   default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RS device: {device}")

    perf   = Walker2DPerfPolicy.load(args.perf_path, attack_path=args.attack_path,
                                     naive_policy=args.naive_policy)
    if args.attack_eps is not None:
        perf.eps = args.attack_eps
    backup = Walker2DBackupPolicy.load(args.backup_path)

    data      = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]

    ckpt     = torch.load(args.switcher_path, map_location="cpu")
    switcher = SwitcherMLP(obs_dim=int(ckpt["obs_dim"]),
                           hidden_dim=int(ckpt["hidden_dim"]))
    switcher.load_state_dict(ckpt["state_dict"])
    switcher.eval()

    rs = VanillaRSSwitcher(switcher, mean, std,
                           sigma=args.sigma,
                           n_samples=args.n_samples,
                           confidence=0.001,
                           device=device)

    controllers = {
        "always_perf":      AlwaysPerfController(perf),
        "always_backup":    AlwaysBackupController(backup),
        "anytime_switcher": AnyTimeSwitcherController(
                                perf, backup, rs,
                                delta_budget_l2=args.delta_budget_l2,
                                detection_k=args.detection_k,
                                recovery_k=args.recovery_k,
                                commit_timeout_k=args.commit_timeout_k,
                                monitoring_delta=args.monitoring_delta),
    }

    print("=== Walker2D burst-attack evaluation ===")
    print(f"episodes={args.episodes}  burst_k={args.burst_k}  "
          f"t_candidate_max={args.t_candidate_max}  "
          f"detection_k={args.detection_k}  recovery_k={args.recovery_k}  "
          f"commit_timeout_k={args.commit_timeout_k}  "
          f"attack_eps={perf.eps}  sigma={args.sigma}  "
          f"delta={args.delta_budget_l2}  n_samples={args.n_samples}  device={device}")
    print()

    all_results = {}
    for name, ctrl in controllers.items():
        print(f"[{name}]")
        for attacked, label in [(False, "clean"), (True, "attacked")]:
            m = run(ctrl, perf, backup, args.episodes, args.seed, attacked,
                    args.burst_k, t_candidate_max=args.t_candidate_max)
            print(f"  {label:8s}  return={m['mean_return']:.1f}±{m['std_return']:.1f}"
                  f"  falls={m['n_fell']}/{args.episodes}"
                  f"  allow_perf={m['mean_allow_perf']:.3f}"
                  f"  R_exec={m['mean_R_exec']:.4f}")
            all_results[f"{name}_{label}"] = m
        print()

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
