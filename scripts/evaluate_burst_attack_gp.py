"""
Evaluate MuJoCo controllers using Gil-Pelaez (GP) certification.

Same evaluation as evaluate_burst_attack_mujoco.py, but uses GPSwitcher instead of
VanillaRSSwitcher. Compares:
  always_perf      — native PPO only
  always_backup    — ATLA only
  anytime_gp       — AnyTimeSwitcherController with GP certification
  anytime_rs       — AnyTimeSwitcherController with MC RS (for comparison)

Also reports per-step certification times to demonstrate GP's speed advantage.

Usage:
    python3.8 scripts/evaluate_burst_attack_gp.py --env hopper \
        --perf-path   Hopper/Hopper_PPO.model \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --backup-path Hopper/Hopper_ATLA.model \
        --gp-switcher-path  models/hopper_switcher_gp.pt \
        --rs-switcher-path  models/hopper_switcher.pt \
        --dataset data/hopper_critical_dataset.npz \
        --sigma 0.1 --n-samples 10000 \
        --delta-budget-l2 0.075 \
        --episodes 30 --seed 0 \
        --burst-k 75 --t-candidate-max 100 \
        --recovery-k 100 --commit-timeout-k 5
"""
import sys
import os
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
    AnyTimeSwitcherController, AdaptiveSwitcherController,
    evaluate_controller,
)
from rs_switcher_common.models import SwitcherMLP, SwitcherDeepMLP
from rs_switcher_common.rs import VanillaRSSwitcher
from rs_switcher_common.gp_models import SwitcherQuadMLP, SwitcherQuadDeepMLP, GPSwitcher


def run(controller, perf, backup, episodes, seed, attacked, burst_k, t_candidate_max,
        attack_norm="linf", attack_eps=None):
    returns, logs = evaluate_controller(
        controller, perf, backup,
        n_episodes=episodes, seed=seed, attack=attacked, burst_k=burst_k,
        t_candidate_max=t_candidate_max,
        attack_norm=attack_norm, attack_eps=attack_eps,
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


def time_certifications(certifier, perf, n_obs=100):
    """Time certification calls on live PPO observations."""
    obs = perf.start_episode()
    times = []
    for _ in range(n_obs):
        t0 = time.perf_counter()
        certifier.certify(obs)
        dt = time.perf_counter() - t0
        times.append(dt)
        action = perf.predict(obs)
        obs, _, done, _ = perf.step(action)
        if done:
            obs = perf.start_episode()
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        choices=list(ENV_REGISTRY.keys()))
    parser.add_argument("--perf-path",        type=str, required=True)
    parser.add_argument("--attack-path",      type=str, required=True)
    parser.add_argument("--backup-path",      type=str, required=True)
    parser.add_argument("--gp-switcher-path", type=str, required=True)
    parser.add_argument("--rs-switcher-path", type=str, default=None,
                        help="If provided, also runs RS switcher for comparison")
    parser.add_argument("--dataset",          type=str, required=True)
    parser.add_argument("--sigma",            type=float, default=0.1)
    parser.add_argument("--n-samples",        type=int, default=10000)
    parser.add_argument("--delta-budget-l2",  type=float, default=0.075)
    parser.add_argument("--episodes",         type=int, default=10)
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--burst-k",          type=int, default=50)
    parser.add_argument("--detection-k",      type=int, default=2)
    parser.add_argument("--recovery-k",       type=int, default=100)
    parser.add_argument("--commit-timeout-k", type=int, default=5)
    parser.add_argument("--recovery-confirm-k", type=int, default=25)
    parser.add_argument("--t-candidate-max",  type=int, default=100)
    parser.add_argument("--monitoring-delta",  type=float, default=None)
    parser.add_argument("--attack-norm",      type=str, default="linf",
                        choices=["linf", "l2"])
    parser.add_argument("--attack-eps",       type=float, default=None)
    parser.add_argument("--device",           type=str, default=None)
    parser.add_argument("--output-json",      type=str, default=None)
    args = parser.parse_args()

    config = ENV_REGISTRY[args.env]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load policies
    perf = MuJoCoPerfPolicy.load(config, args.perf_path,
                                  attack_path=args.attack_path)
    backup = MuJoCoBackupPolicy.load(config, args.backup_path)

    data = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]

    # Load GP switcher
    gp_ck = torch.load(args.gp_switcher_path, map_location="cpu")
    if gp_ck.get("model_type") == "quad_deep":
        gp_model = SwitcherQuadDeepMLP(obs_dim=int(gp_ck["obs_dim"]),
                                        backbone_dims=gp_ck["backbone_dims"])
    else:
        gp_model = SwitcherQuadMLP(obs_dim=int(gp_ck["obs_dim"]),
                                    hidden_dim=int(gp_ck["hidden_dim"]))
    gp_model.load_state_dict(gp_ck["state_dict"])
    gp_model.eval()
    gp_cert = GPSwitcher(gp_model, mean, std, sigma=args.sigma, device="cpu")

    # Build controllers
    controllers = {
        "always_perf":   AlwaysPerfController(perf),
        "always_backup": AlwaysBackupController(backup),
        "anytime_gp":    AnyTimeSwitcherController(
                             perf, backup, gp_cert,
                             delta_budget_l2=args.delta_budget_l2,
                             detection_k=args.detection_k,
                             recovery_k=args.recovery_k,
                             commit_timeout_k=args.commit_timeout_k,
                             monitoring_delta=args.monitoring_delta),
        "adaptive_gp":   AdaptiveSwitcherController(
                             perf, backup, gp_cert,
                             delta_budget_l2=args.delta_budget_l2,
                             detection_k=args.detection_k,
                             recovery_confirm_k=args.recovery_confirm_k,
                             commit_timeout_k=args.commit_timeout_k,
                             monitoring_delta=args.monitoring_delta),
    }

    # Optionally add RS for comparison
    rs_cert = None
    if args.rs_switcher_path:
        rs_ck = torch.load(args.rs_switcher_path, map_location="cpu")
        if "hidden_dims" in rs_ck:
            rs_model = SwitcherDeepMLP(obs_dim=int(rs_ck["obs_dim"]),
                                       hidden_dims=rs_ck["hidden_dims"])
        else:
            rs_model = SwitcherMLP(obs_dim=int(rs_ck["obs_dim"]),
                                    hidden_dim=int(rs_ck["hidden_dim"]))
        rs_model.load_state_dict(rs_ck["state_dict"])
        rs_model.eval()
        rs_cert = VanillaRSSwitcher(rs_model, mean, std,
                                     sigma=args.sigma,
                                     n_samples=args.n_samples,
                                     confidence=0.001,
                                     device=device)
        controllers["anytime_rs"] = AnyTimeSwitcherController(
                                        perf, backup, rs_cert,
                                        delta_budget_l2=args.delta_budget_l2,
                                        detection_k=args.detection_k,
                                        recovery_k=args.recovery_k,
                                        commit_timeout_k=args.commit_timeout_k,
                                        monitoring_delta=args.monitoring_delta)
        controllers["adaptive_rs"] = AdaptiveSwitcherController(
                                        perf, backup, rs_cert,
                                        delta_budget_l2=args.delta_budget_l2,
                                        detection_k=args.detection_k,
                                        recovery_confirm_k=args.recovery_confirm_k,
                                        commit_timeout_k=args.commit_timeout_k,
                                        monitoring_delta=args.monitoring_delta)

    # --- Timing comparison ---
    print(f"=== {config.name} GP vs RS Timing ===")
    print("Timing GP certification on 200 live observations...")
    gp_times = time_certifications(gp_cert, perf, n_obs=200)
    print(f"  GP: mean={np.mean(gp_times)*1000:.2f} ms, "
          f"median={np.median(gp_times)*1000:.2f} ms, "
          f"p95={np.percentile(gp_times, 95)*1000:.2f} ms")

    if rs_cert is not None:
        print(f"Timing RS certification (n_samples={args.n_samples}) on 200 live observations...")
        rs_times = time_certifications(rs_cert, perf, n_obs=200)
        print(f"  RS: mean={np.mean(rs_times)*1000:.2f} ms, "
              f"median={np.median(rs_times)*1000:.2f} ms, "
              f"p95={np.percentile(rs_times, 95)*1000:.2f} ms")
        speedup = np.mean(rs_times) / np.mean(gp_times)
        print(f"  Speedup: {speedup:.1f}x")
    print()

    # --- Full evaluation ---
    print(f"=== {config.name} Burst-Attack Evaluation ===")
    print(f"episodes={args.episodes}  burst_k={args.burst_k}  "
          f"t_candidate_max={args.t_candidate_max}  "
          f"sigma={args.sigma}  delta={args.delta_budget_l2}")
    print()

    all_results = {}
    for name, ctrl in controllers.items():
        print(f"[{name}]")
        t0 = time.time()
        for attacked, label in [(False, "clean"), (True, "attacked")]:
            m = run(ctrl, perf, backup, args.episodes, args.seed, attacked,
                    args.burst_k, t_candidate_max=args.t_candidate_max,
                    attack_norm=args.attack_norm, attack_eps=args.attack_eps)
            print(f"  {label:8s}  return={m['mean_return']:.1f}+/-{m['std_return']:.1f}"
                  f"  falls={m['n_fell']}/{args.episodes}"
                  f"  allow_perf={m['mean_allow_perf']:.3f}"
                  f"  R_exec={m['mean_R_exec']:.4f}")
            all_results[f"{name}_{label}"] = m
        wall = time.time() - t0
        print(f"  (wall time: {wall:.1f}s)")
        print()

    # --- Timing summary ---
    print("=== Timing Summary ===")
    print(f"  GP cert/obs:  {np.mean(gp_times)*1000:.2f} ms (mean), "
          f"{np.percentile(gp_times, 95)*1000:.2f} ms (p95)")
    if rs_cert is not None:
        print(f"  RS cert/obs:  {np.mean(rs_times)*1000:.2f} ms (mean), "
              f"{np.percentile(rs_times, 95)*1000:.2f} ms (p95)")
        print(f"  GP is {speedup:.1f}x faster than RS")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        timing = {
            "gp_mean_ms": float(np.mean(gp_times) * 1000),
            "gp_p95_ms": float(np.percentile(gp_times, 95) * 1000),
        }
        if rs_cert is not None:
            timing["rs_mean_ms"] = float(np.mean(rs_times) * 1000)
            timing["rs_p95_ms"] = float(np.percentile(rs_times, 95) * 1000)
            timing["speedup"] = float(speedup)
        all_results["timing"] = timing
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
