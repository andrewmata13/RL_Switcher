"""
Evaluate ContinuousSwitcherController under multi-burst and arbitrary attacks.

Compares:
  always_perf         — native PPO (no defense)
  always_backup       — ATLA only (conservative)
  adaptive_gp         — AdaptiveSwitcherController (single-burst design)
  continuous_gp       — ContinuousSwitcherController (arbitrary attack design)

Attack modes:
  single    — one burst per episode (baseline comparison)
  multi     — n_bursts bursts of burst_k steps with cooldown_k gap
  arbitrary — each step independently attacked (Bernoulli)

Usage:
    python3.8 scripts/evaluate_continuous_controller.py --env hopper \
        --perf-path   Hopper/Hopper_PPO.model \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --backup-path Hopper/Hopper_ATLA.model \
        --gp-switcher-path models/hopper_switcher_gp_h512.pt \
        --dataset data/hopper_critical_dataset.npz \
        --sigma 0.1 --delta-budget-l2 0.075 \
        --episodes 30 --seed 0 \
        --attack-mode multi --n-bursts 3 --burst-k 50 --cooldown-k 50 \
        --K-enter 3 --K-exit 10 \
        --attack-norm l2 --attack-eps 0.13 \
        --output-json results/hopper_continuous_multi.json
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
import numpy as np
import torch

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.controllers import MuJoCoPerfPolicy, MuJoCoBackupPolicy
from rs_switcher_common.evaluation import (
    AlwaysPerfController, AlwaysBackupController,
    AdaptiveSwitcherController, ContinuousSwitcherController,
    evaluate_controller,
)
from rs_switcher_common.gp_models import load_gp_switcher, GPSwitcher
from rs_switcher_common.models import load_switcher
from rs_switcher_common.rs import VanillaRSSwitcher


def run(controller, perf, backup, args, attacked):
    returns, logs = evaluate_controller(
        controller, perf, backup,
        n_episodes=args.episodes, seed=args.seed,
        attack=attacked, burst_k=args.burst_k,
        t_candidate_max=args.t_candidate_max,
        attack_norm=args.attack_norm, attack_eps=args.attack_eps,
        attack_mode=args.attack_mode,
        n_bursts=args.n_bursts, cooldown_k=args.cooldown_k,
    )
    allow_means = [l["allow_perf"] for l in logs]
    n_fell = sum(l["fell"] for l in logs)
    atk_steps = [l.get("attacked_steps", 0) for l in logs]
    return {
        "mean_return":     float(np.mean(returns)),
        "std_return":      float(np.std(returns)),
        "mean_allow_perf": float(np.mean(allow_means)),
        "n_fell":          n_fell,
        "fall_rate":       n_fell / len(logs),
        "mean_attacked_steps": float(np.mean(atk_steps)),
        "returns":         [float(r) for r in returns],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--perf-path", required=True)
    p.add_argument("--attack-path", required=True)
    p.add_argument("--backup-path", required=True)
    p.add_argument("--gp-switcher-path", default=None)
    p.add_argument("--rs-switcher-path", default=None,
                   help="RS switcher checkpoint (SwitcherRobustMLP/DeepMLP/MLP). "
                        "If given, used instead of GP certifier.")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="RS Monte Carlo samples per certification call")
    p.add_argument("--dataset", required=True)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--delta-budget-l2", type=float, default=0.075)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)

    # Attack params
    p.add_argument("--attack-mode", default="multi",
                   choices=["single", "multi", "arbitrary"])
    p.add_argument("--burst-k", type=int, default=50)
    p.add_argument("--n-bursts", type=int, default=3)
    p.add_argument("--cooldown-k", type=int, default=50)
    p.add_argument("--t-candidate-max", type=int, default=100)
    p.add_argument("--attack-norm", default="linf", choices=["linf", "l2"])
    p.add_argument("--attack-eps", type=float, default=None)

    # Adaptive controller params (for comparison)
    p.add_argument("--detection-k", type=int, default=2)
    p.add_argument("--recovery-confirm-k", type=int, default=10)
    p.add_argument("--commit-timeout-k", type=int, default=5)
    p.add_argument("--monitoring-delta", type=float, default=None)

    # Continuous controller params
    p.add_argument("--K-enter", type=int, default=3)
    p.add_argument("--K-exit", type=int, default=10)
    p.add_argument("--forgive-decay", type=int, default=1)

    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    config = ENV_REGISTRY[args.env]
    perf = MuJoCoPerfPolicy.load(config, args.perf_path,
                                  attack_path=args.attack_path)
    backup = MuJoCoBackupPolicy.load(config, args.backup_path)

    data = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]

    # Load certifier — RS switcher takes priority over GP if both provided
    if args.rs_switcher_path is not None:
        rs_ck = torch.load(args.rs_switcher_path, map_location="cpu")
        rs_model = load_switcher(rs_ck)
        cert = VanillaRSSwitcher(rs_model, mean, std, sigma=args.sigma,
                                 n_samples=args.n_samples, device="cpu")
        cert_label = "rs"
    elif args.gp_switcher_path is not None:
        gp_ck = torch.load(args.gp_switcher_path, map_location="cpu")
        gp_model = load_gp_switcher(gp_ck)
        cert = GPSwitcher(gp_model, mean, std, sigma=args.sigma, device="cpu")
        cert_label = "gp"
    else:
        raise ValueError("Must provide --gp-switcher-path or --rs-switcher-path")

    controllers = {
        "always_perf":          AlwaysPerfController(perf),
        "always_backup":        AlwaysBackupController(backup),
        f"adaptive_{cert_label}": AdaptiveSwitcherController(
                                    perf, backup, cert,
                                    delta_budget_l2=args.delta_budget_l2,
                                    detection_k=args.detection_k,
                                    recovery_confirm_k=args.recovery_confirm_k,
                                    commit_timeout_k=args.commit_timeout_k,
                                    monitoring_delta=args.monitoring_delta),
        f"continuous_{cert_label}": ContinuousSwitcherController(
                                    perf, backup, cert,
                                    delta_budget_l2=args.delta_budget_l2,
                                    K_enter=args.K_enter,
                                    K_exit=args.K_exit,
                                    monitoring_delta=args.monitoring_delta,
                                    forgive_decay=args.forgive_decay),
    }

    print(f"=== {config.name} Continuous Controller Evaluation ===")
    print(f"attack_mode={args.attack_mode}  burst_k={args.burst_k}  "
          f"n_bursts={args.n_bursts}  cooldown_k={args.cooldown_k}")
    print(f"K_enter={args.K_enter}  K_exit={args.K_exit}  "
          f"forgive_decay={args.forgive_decay}")
    print(f"sigma={args.sigma}  delta={args.delta_budget_l2}  "
          f"attack_norm={args.attack_norm}  attack_eps={args.attack_eps}")
    print(f"episodes={args.episodes}  seed={args.seed}")
    print()

    all_results = {}
    for name, ctrl in controllers.items():
        print(f"[{name}]")
        t0 = time.time()
        for attacked, label in [(False, "clean"), (True, "attacked")]:
            m = run(ctrl, perf, backup, args, attacked)
            atk_info = (f"  atk_steps={m['mean_attacked_steps']:.0f}"
                        if attacked else "")
            print(f"  {label:8s}  return={m['mean_return']:.1f}+/-{m['std_return']:.1f}"
                  f"  falls={m['n_fell']}/{args.episodes}"
                  f"  PPO%={m['mean_allow_perf']:.3f}{atk_info}")
            all_results[f"{name}_{label}"] = m
        wall = time.time() - t0
        print(f"  (wall: {wall:.1f}s)")
        print()

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        all_results["config"] = {
            "env": args.env, "attack_mode": args.attack_mode,
            "burst_k": args.burst_k, "n_bursts": args.n_bursts,
            "cooldown_k": args.cooldown_k, "K_enter": args.K_enter,
            "K_exit": args.K_exit, "forgive_decay": args.forgive_decay,
            "sigma": args.sigma, "delta_budget_l2": args.delta_budget_l2,
            "attack_norm": args.attack_norm, "attack_eps": args.attack_eps,
        }
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
