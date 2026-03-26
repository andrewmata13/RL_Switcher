"""
Evaluate MuJoCo controllers under the Zhang et al. adversarial attack.

Compares three controllers:
  always_perf      — native PPO only
  always_backup    — ATLA only
  anytime_switcher — RS-certified detection + ATLA recovery + permanent PPO commit

Usage:
    python3.8 scripts/evaluate_burst_attack_mujoco.py --env hopper \
        --perf-path   Hopper/Hopper_PPO.model \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --backup-path Hopper/Hopper_ATLA.model \
        --switcher-path models/hopper_switcher.pt \
        --dataset  data/hopper_critical_dataset.npz \
        --sigma 0.1 --n-samples 10000 \
        --delta-budget-l2 0.075 \
        --episodes 30 --seed 0 \
        --burst-k 50 --t-candidate-max 300 \
        --recovery-k 100 --commit-timeout-k 5 \
        --device cuda
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import torch

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.controllers import MuJoCoPerfPolicy, MuJoCoBackupPolicy
from rs_switcher_common.evaluation import (
    AlwaysPerfController, AlwaysBackupController,
    AnyTimeSwitcherController, AdaptiveSwitcherController,
    evaluate_controller,
)
from rs_switcher_common.models import SwitcherMLP, SwitcherDeepMLP
from rs_switcher_common.rs import VanillaRSSwitcher


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
    parser.add_argument("--env", type=str, required=True,
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment name")
    parser.add_argument("--perf-path",       type=str, required=True)
    parser.add_argument("--attack-path",     type=str, required=True)
    parser.add_argument("--backup-path",     type=str, required=True)
    parser.add_argument("--switcher-path",   type=str, required=True)
    parser.add_argument("--dataset",         type=str, required=True)
    parser.add_argument("--sigma",           type=float, default=0.1)
    parser.add_argument("--n-samples",       type=int,   default=10000,
                        help="MC samples for RS certification (use 10000 with GPU)")
    parser.add_argument("--delta-budget-l2", type=float, default=0.075,
                        help="RS threshold for both detection and commit gate.")
    parser.add_argument("--episodes",        type=int,   default=10)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--burst-k",         type=int,   default=50)
    parser.add_argument("--detection-k",     type=int,   default=2,
                        help="Consecutive RS failures before declaring attack")
    parser.add_argument("--recovery-k",      type=int,   default=100,
                        help="ATLA recovery steps after attack detected (>= burst_k)")
    parser.add_argument("--commit-timeout-k",type=int,   default=5,
                        help="RS-check steps before forced PPO commit after recovery")
    parser.add_argument("--recovery-confirm-k", type=int, default=25,
                        help="Consecutive certified-safe steps in ATLA before exit (adaptive controller). "
                             "Recommended: hopper=25, halfcheetah=3, walker2d=5")
    parser.add_argument("--t-candidate-max", type=int,   default=100,
                        help="Max step at which burst attack can start")
    parser.add_argument("--device",          type=str,   default=None,
                        help="Torch device for RS (default: cuda if available, else cpu)")
    parser.add_argument("--monitoring-delta", type=float, default=None,
                        help="Phase-1 detection threshold (default: same as delta-budget-l2). "
                             "Set to 0.0 to use pred-only detection.")
    parser.add_argument("--naive-policy",    action="store_true")
    parser.add_argument("--attack-eps",      type=float, default=None)
    parser.add_argument("--output-json",     type=str,   default=None,
                        help="If set, write results dict to this JSON file")
    args = parser.parse_args()

    config = ENV_REGISTRY[args.env]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RS device: {device}")

    perf = MuJoCoPerfPolicy.load(config, args.perf_path,
                                  attack_path=args.attack_path,
                                  naive_policy=args.naive_policy)
    if args.attack_eps is not None:
        perf.eps = args.attack_eps
    backup = MuJoCoBackupPolicy.load(config, args.backup_path)

    data      = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]

    ckpt = torch.load(args.switcher_path, map_location="cpu")
    if "hidden_dims" in ckpt:
        switcher = SwitcherDeepMLP(obs_dim=int(ckpt["obs_dim"]),
                                    hidden_dims=ckpt["hidden_dims"])
    else:
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
        "adaptive_switcher": AdaptiveSwitcherController(
                                perf, backup, rs,
                                delta_budget_l2=args.delta_budget_l2,
                                detection_k=args.detection_k,
                                recovery_confirm_k=args.recovery_confirm_k,
                                commit_timeout_k=args.commit_timeout_k,
                                monitoring_delta=args.monitoring_delta),
    }

    print(f"=== {config.name} burst-attack evaluation ===")
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
            print(f"  {label:8s}  return={m['mean_return']:.1f}+/-{m['std_return']:.1f}"
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
