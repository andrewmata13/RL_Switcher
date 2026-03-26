"""
Worst-case burst-start evaluation across MuJoCo environments.

Phase 1 -- T-sweep (pilot): sweep T_candidate in [0, t_candidate_max] with a
           fixed stride.  Run pilot_eps episodes per T using seed=0.
           The worst-case T* minimises the adaptive_switcher return.
Phase 2 -- Final eval: run seeds episodes at the fixed worst-case T*.
           Report all controllers (always_perf, always_backup,
           anytime_switcher, adaptive_switcher) both clean and attacked.

Usage:
    python3.8 scripts/evaluate_worst_case_start.py --env hopper \
        --perf-path   Hopper/Hopper_PPO.model \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --backup-path Hopper/Hopper_ATLA.model \
        --switcher-path models/hopper_switcher.pt \
        --dataset data/hopper_critical_dataset.npz \
        --sigma 0.1 --n-samples 1000 \
        --delta-budget-l2 0.075 \
        --burst-k 75 --t-candidate-max 300 \
        --seeds 30 --device cpu
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

# Per-env recommended defaults (from adaptive_control branch tuning)
ENV_DEFAULTS = {
    "hopper": dict(
        sigma=0.1, delta=0.075, burst_k=75, t_max=300,
        detection_k=2, recovery_k=100, recovery_confirm_k=25, commit_timeout_k=5,
    ),
    "halfcheetah": dict(
        sigma=0.2, delta=0.15, burst_k=100, t_max=100,
        detection_k=5, recovery_k=100, recovery_confirm_k=3, commit_timeout_k=5,
    ),
    "walker2d": dict(
        sigma=0.05, delta=0.05, burst_k=100, t_max=100,
        detection_k=3, recovery_k=100, recovery_confirm_k=5, commit_timeout_k=5,
    ),
}


def run_fixed_T(controller, perf, backup, burst_k, t_candidate_max,
                n_episodes, seed, attacked, T_fixed):
    returns, logs = evaluate_controller(
        controller, perf, backup,
        n_episodes=n_episodes, seed=seed, attack=attacked,
        burst_k=burst_k, t_candidate_max=t_candidate_max,
        t_candidate_fixed=T_fixed,
    )
    allow_means = [l["allow_perf"] for l in logs]
    R_means     = [l["R_exec"] for l in logs if not np.isnan(l["R_exec"])]
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
                        choices=list(ENV_REGISTRY.keys()))
    parser.add_argument("--perf-path",       type=str, required=True)
    parser.add_argument("--attack-path",     type=str, required=True)
    parser.add_argument("--backup-path",     type=str, required=True)
    parser.add_argument("--switcher-path",   type=str, required=True)
    parser.add_argument("--dataset",         type=str, required=True)
    parser.add_argument("--sigma",           type=float, default=None)
    parser.add_argument("--n-samples",       type=int, default=1000)
    parser.add_argument("--delta-budget-l2", type=float, default=None)
    parser.add_argument("--burst-k",         type=int, default=None)
    parser.add_argument("--t-candidate-max", type=int, default=None)
    parser.add_argument("--detection-k",     type=int, default=None)
    parser.add_argument("--recovery-k",      type=int, default=None)
    parser.add_argument("--recovery-confirm-k", type=int, default=None)
    parser.add_argument("--commit-timeout-k", type=int, default=None)
    parser.add_argument("--monitoring-delta", type=float, default=None)
    parser.add_argument("--pilot-eps",       type=int, default=5)
    parser.add_argument("--t-stride",        type=int, default=None)
    parser.add_argument("--seeds",           type=int, default=30)
    parser.add_argument("--device",          type=str, default=None)
    parser.add_argument("--output-json",     type=str, default=None)
    args = parser.parse_args()

    config = ENV_REGISTRY[args.env]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Apply per-env defaults for any unset parameters
    defaults = ENV_DEFAULTS.get(args.env, {})
    if args.sigma is None:           args.sigma = defaults.get("sigma", 0.1)
    if args.delta_budget_l2 is None: args.delta_budget_l2 = defaults.get("delta", 0.075)
    if args.burst_k is None:         args.burst_k = defaults.get("burst_k", 75)
    if args.t_candidate_max is None: args.t_candidate_max = defaults.get("t_max", 300)
    if args.detection_k is None:     args.detection_k = defaults.get("detection_k", 2)
    if args.recovery_k is None:      args.recovery_k = defaults.get("recovery_k", 100)
    if args.recovery_confirm_k is None: args.recovery_confirm_k = defaults.get("recovery_confirm_k", 25)
    if args.commit_timeout_k is None: args.commit_timeout_k = defaults.get("commit_timeout_k", 5)

    print(f"RS device: {device}  env={args.env}")

    perf = MuJoCoPerfPolicy.load(config, args.perf_path,
                                  attack_path=args.attack_path)
    backup = MuJoCoBackupPolicy.load(config, args.backup_path)

    data = np.load(args.dataset)
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

    t_stride = args.t_stride or max(1, args.t_candidate_max // 10)
    T_values = list(range(0, args.t_candidate_max + 1, t_stride))

    # -- Phase 1: T-sweep (pilot) ---------------------------------------------
    print(f"\n=== Phase 1: T-sweep (pilot_eps={args.pilot_eps}, stride={t_stride}) ===")
    print(f"T values: {T_values}")
    adaptive_ctrl = controllers["adaptive_switcher"]
    perf_ctrl     = controllers["always_perf"]

    sweep_results = {}
    for T in T_values:
        r_adapt = run_fixed_T(adaptive_ctrl, perf, backup, args.burst_k,
                              args.t_candidate_max, args.pilot_eps,
                              seed=0, attacked=True, T_fixed=T)
        r_perf  = run_fixed_T(perf_ctrl, perf, backup, args.burst_k,
                              args.t_candidate_max, args.pilot_eps,
                              seed=0, attacked=True, T_fixed=T)
        sweep_results[T] = {"adaptive": r_adapt["mean_return"],
                            "perf":     r_perf["mean_return"]}
        print(f"  T={T:4d}  always_perf={r_perf['mean_return']:.1f}"
              f"  adaptive={r_adapt['mean_return']:.1f}"
              f"  falls={r_adapt['n_fell']}/{args.pilot_eps}")

    T_star = min(sweep_results, key=lambda T: sweep_results[T]["adaptive"])
    print(f"\nWorst-case T* = {T_star}  "
          f"(adaptive return = {sweep_results[T_star]['adaptive']:.1f})")

    # -- Phase 2: Final eval at T* across all seeds ----------------------------
    seed_list = list(range(args.seeds))
    print(f"\n=== Phase 2: Final eval at T*={T_star}, {args.seeds} seeds ===")
    print(f"burst_k={args.burst_k}  sigma={args.sigma}  delta={args.delta_budget_l2}")
    print()

    all_results = {"T_star": T_star, "sweep": {str(k): v for k, v in sweep_results.items()}}

    for name, ctrl in controllers.items():
        print(f"[{name}]")
        for attacked, label in [(False, "clean"), (True, "attacked")]:
            per_seed = []
            for s in seed_list:
                m = run_fixed_T(ctrl, perf, backup, args.burst_k,
                                args.t_candidate_max, n_episodes=1, seed=s,
                                attacked=attacked,
                                T_fixed=T_star if attacked else None)
                per_seed.append(m)
            returns    = [m["mean_return"] for m in per_seed]
            n_fell     = sum(m["n_fell"] for m in per_seed)
            allow_mean = np.mean([m["mean_allow_perf"] for m in per_seed])
            R_vals     = [m["mean_R_exec"] for m in per_seed
                          if not np.isnan(m["mean_R_exec"])]
            R_mean     = float(np.nanmean(R_vals)) if R_vals else float("nan")
            print(f"  {label:8s}  return={np.mean(returns):.1f}+/-{np.std(returns):.1f}"
                  f"  falls={n_fell}/{args.seeds}"
                  f"  allow_perf={allow_mean:.3f}"
                  f"  R_exec={R_mean:.4f}")
            all_results[f"{name}_{label}"] = {
                "mean_return": float(np.mean(returns)),
                "std_return":  float(np.std(returns)),
                "n_fell":      n_fell,
                "fall_rate":   n_fell / args.seeds,
                "mean_allow_perf": float(allow_mean),
                "mean_R_exec": R_mean,
                "returns": returns,
            }
        print()

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
