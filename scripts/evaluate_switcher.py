import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from cartpole_ags_rs_switcher.controllers import PerfPolicy, QuantizedLQRBackup
from cartpole_ags_rs_switcher.evaluation import CertifiedSwitcherController, evaluate_controller
from cartpole_ags_rs_switcher.models import SwitcherMLP
from cartpole_ags_rs_switcher.rs import VanillaRSSwitcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-path", type=str, required=True)
    parser.add_argument("--switcher-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--delta-budget-l2", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perf = PerfPolicy.load(args.perf_path, device=device)
    backup = QuantizedLQRBackup()

    data = np.load(args.dataset)
    mean = data["state_mean"]
    std = data["state_std"]

    ckpt = torch.load(args.switcher_path, map_location="cpu")
    switcher = SwitcherMLP(obs_dim=int(ckpt["obs_dim"]), hidden_dim=int(ckpt["hidden_dim"]))
    switcher.load_state_dict(ckpt["state_dict"])
    switcher.eval()

    rs = VanillaRSSwitcher(switcher, mean, std, sigma=float(args.sigma),
                           n_samples=args.n_samples, confidence=args.confidence)

    print(f"delta_budget_l2={args.delta_budget_l2} sigma={args.sigma} n_samples={args.n_samples}")
    controller = CertifiedSwitcherController(perf, backup, rs, delta_budget_l2=float(args.delta_budget_l2))
    returns, logs = evaluate_controller("CartPole-v1", controller, episodes=args.episodes)
    print(f"mean return={np.mean(returns):.2f} std={np.std(returns):.2f}")
    print(f"mean allow_perf={np.nanmean([x['allow_perf'] for x in logs]):.3f}")
    print(f"mean R_exec={np.nanmean([x['R_exec'] for x in logs]):.4f}")


if __name__ == "__main__":
    main()
