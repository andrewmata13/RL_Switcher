"""
Sweep K_enter, K_exit, delta_budget_l2 for ContinuousSwitcherController (GP).
Only runs continuous controller + baselines to save time.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json, time
import numpy as np
import torch
from itertools import product

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.controllers import MuJoCoPerfPolicy, MuJoCoBackupPolicy
from rs_switcher_common.evaluation import (
    AlwaysPerfController, AlwaysBackupController,
    ContinuousSwitcherController, evaluate_controller,
)
from rs_switcher_common.gp_models import load_gp_switcher, GPSwitcher


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
    return {
        "median_return": float(np.median(returns)),
        "mean_return":   float(np.mean(returns)),
        "std_return":    float(np.std(returns)),
        "ppo_pct":       float(np.mean(allow_means)),
        "n_fell":        n_fell,
        "returns":       [float(r) for r in returns],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="halfcheetah")
    p.add_argument("--perf-path", required=True)
    p.add_argument("--attack-path", required=True)
    p.add_argument("--backup-path", required=True)
    p.add_argument("--gp-switcher-path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    # Attack params
    p.add_argument("--attack-mode", default="multi")
    p.add_argument("--burst-k", type=int, default=100)
    p.add_argument("--n-bursts", type=int, default=3)
    p.add_argument("--cooldown-k", type=int, default=100)
    p.add_argument("--t-candidate-max", type=int, default=100)
    p.add_argument("--attack-norm", default="l2")
    p.add_argument("--attack-eps", type=float, default=0.5)

    # Sweep ranges
    p.add_argument("--deltas", type=str, default="0.05,0.1,0.2")
    p.add_argument("--K-enters", type=str, default="2,3,5")
    p.add_argument("--K-exits", type=str, default="3,5,10")
    p.add_argument("--forgive-decays", type=str, default="1.0")

    p.add_argument("--output-json", default="results/halfcheetah_gp_sweep.json")
    args = p.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    K_enters = [int(x) for x in args.K_enters.split(",")]
    K_exits = [int(x) for x in args.K_exits.split(",")]
    forgives = [float(x) for x in args.forgive_decays.split(",")]

    config = ENV_REGISTRY[args.env]
    perf = MuJoCoPerfPolicy.load(config, args.perf_path, attack_path=args.attack_path)
    backup = MuJoCoBackupPolicy.load(config, args.backup_path)

    data = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]
    gp_ck = torch.load(args.gp_switcher_path, map_location="cpu")
    gp_model = load_gp_switcher(gp_ck)
    cert = GPSwitcher(gp_model, mean, std, sigma=args.sigma, device="cpu")

    # Run baselines once
    print(f"=== Baselines ({args.episodes} episodes) ===")
    baselines = {}
    for name, ctrl in [("always_ppo", AlwaysPerfController(perf)),
                       ("always_atla", AlwaysBackupController(backup))]:
        for attacked, label in [(False, "clean"), (True, "attacked")]:
            m = run(ctrl, perf, backup, args, attacked)
            key = f"{name}_{label}"
            baselines[key] = m
            print(f"  {key:25s}  median={m['median_return']:.0f}  "
                  f"mean={m['mean_return']:.0f}+/-{m['std_return']:.0f}  "
                  f"falls={m['n_fell']}")

    # Sweep
    grid = list(product(deltas, K_enters, K_exits, forgives))
    print(f"\n=== Sweep: {len(grid)} configs ===\n")
    sweep_results = []

    for i, (delta, ke, kx, fd) in enumerate(grid):
        ctrl = ContinuousSwitcherController(
            perf, backup, cert,
            delta_budget_l2=delta, K_enter=ke, K_exit=kx,
            forgive_decay=fd,
        )
        row = {"delta": delta, "K_enter": ke, "K_exit": kx, "forgive_decay": fd}
        for attacked, label in [(False, "clean"), (True, "attacked")]:
            m = run(ctrl, perf, backup, args, attacked)
            row[f"{label}_median"] = m["median_return"]
            row[f"{label}_mean"] = m["mean_return"]
            row[f"{label}_std"] = m["std_return"]
            row[f"{label}_ppo_pct"] = m["ppo_pct"]
            row[f"{label}_falls"] = m["n_fell"]
            row[f"{label}_returns"] = m["returns"]

        print(f"[{i+1}/{len(grid)}] delta={delta} Ke={ke} Kx={kx} fd={fd}  "
              f"clean={row['clean_median']:.0f} (PPO {row['clean_ppo_pct']:.1%})  "
              f"attacked={row['attacked_median']:.0f} (PPO {row['attacked_ppo_pct']:.1%})")
        sweep_results.append(row)

    # Save
    out = {"baselines": baselines, "sweep": sweep_results,
           "config": {"env": args.env, "sigma": args.sigma,
                      "attack_mode": args.attack_mode, "burst_k": args.burst_k,
                      "n_bursts": args.n_bursts, "cooldown_k": args.cooldown_k,
                      "attack_norm": args.attack_norm, "attack_eps": args.attack_eps,
                      "episodes": args.episodes, "seed": args.seed}}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output_json}")

    # Print top 5 by attacked median
    print("\n=== Top 5 by attacked median ===")
    sweep_results.sort(key=lambda r: r["attacked_median"], reverse=True)
    for r in sweep_results[:5]:
        print(f"  delta={r['delta']} Ke={r['K_enter']} Kx={r['K_exit']} fd={r['forgive_decay']}  "
              f"clean={r['clean_median']:.0f} (PPO {r['clean_ppo_pct']:.1%})  "
              f"attacked={r['attacked_median']:.0f} (PPO {r['attacked_ppo_pct']:.1%})")


if __name__ == "__main__":
    main()
