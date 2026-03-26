"""
Compare Gil-Pelaez (GP) vs Randomized Smoothing (RS) certification on the Hopper switcher.

Compares:
  1. Certified radius (R) on the same observations
  2. Wall-clock time per certification
  3. Agreement rate (do both methods agree on prediction?)
  4. Accuracy on the labeled dataset

Usage:
    python3.8 scripts/compare_gp_vs_rs.py \
        --dataset data/hopper_critical_dataset.npz \
        --rs-switcher-path models/hopper_switcher.pt \
        --gp-switcher-path models/hopper_switcher_gp.pt \
        --sigma 0.1 \
        --n-samples 10000 \
        --n-obs 200 \
        --device cuda
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import numpy as np
import torch

from rs_switcher_common.models import SwitcherMLP, SwitcherDeepMLP
from rs_switcher_common.rs import VanillaRSSwitcher
from rs_switcher_common.gp_models import SwitcherQuadMLP, GPSwitcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",            type=str, required=True)
    parser.add_argument("--rs-switcher-path",   type=str, required=True,
                        help="Path to RS switcher (SwitcherMLP, 1-logit)")
    parser.add_argument("--gp-switcher-path",   type=str, required=True,
                        help="Path to GP switcher (SwitcherQuadMLP, 2-class)")
    parser.add_argument("--sigma",              type=float, default=0.1)
    parser.add_argument("--n-samples",          type=int, default=10000,
                        help="MC samples for RS (default 10000)")
    parser.add_argument("--n-obs",              type=int, default=200,
                        help="Number of observations to certify")
    parser.add_argument("--device",             type=str, default=None)
    parser.add_argument("--output-json",        type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    mean, std = data["state_mean"], data["state_std"]
    print(f"Dataset: {len(X)} samples, obs_dim={X.shape[1]}")

    # Load RS switcher
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

    # Load GP switcher
    gp_ck = torch.load(args.gp_switcher_path, map_location="cpu")
    gp_model = SwitcherQuadMLP(obs_dim=int(gp_ck["obs_dim"]),
                                hidden_dim=int(gp_ck["hidden_dim"]))
    gp_model.load_state_dict(gp_ck["state_dict"])
    gp_model.eval()

    gp_cert = GPSwitcher(gp_model, mean, std, sigma=args.sigma, device="cpu")

    # Select observations to certify (balanced: half clean, half adversarial)
    n = min(args.n_obs, len(X))
    idx_clean = np.where(y == 0)[0]
    idx_adv = np.where(y == 1)[0]
    n_each = n // 2
    np.random.seed(42)
    idx = np.concatenate([
        np.random.choice(idx_clean, min(n_each, len(idx_clean)), replace=False),
        np.random.choice(idx_adv, min(n_each, len(idx_adv)), replace=False),
    ])
    np.random.shuffle(idx)
    print(f"Certifying {len(idx)} observations ({n_each} clean, {n_each} adversarial)")
    print()

    # --- RS certification ---
    print(f"=== RS Certification (n_samples={args.n_samples}, device={device}) ===")
    rs_results = []
    rs_total_time = 0.0
    for i, ix in enumerate(idx):
        obs = X[ix]
        t0 = time.perf_counter()
        pred, pA, R = rs_cert.certify(obs)
        dt = time.perf_counter() - t0
        rs_results.append({"pred": pred, "pA": pA, "R": R, "time": dt, "label": int(y[ix])})
        rs_total_time += dt

    rs_radii = [r["R"] for r in rs_results]
    rs_times = [r["time"] for r in rs_results]
    rs_preds = [r["pred"] for r in rs_results]
    rs_correct = sum(1 for r in rs_results if r["pred"] == r["label"]) / len(rs_results)

    print(f"  Avg time/obs: {np.mean(rs_times)*1000:.2f} ms")
    print(f"  Avg radius:   {np.mean(rs_radii):.4f}")
    print(f"  Median radius: {np.median(rs_radii):.4f}")
    print(f"  Accuracy:     {rs_correct:.3f}")
    print(f"  Total time:   {rs_total_time:.2f} s")
    print()

    # --- GP certification ---
    print(f"=== GP Certification (single-pass, device=cpu) ===")
    gp_results = []
    gp_total_time = 0.0
    for i, ix in enumerate(idx):
        obs = X[ix]
        pred, pA, R, dt = gp_cert.certify_timed(obs)
        gp_results.append({"pred": pred, "pA": pA, "R": R, "time": dt, "label": int(y[ix])})
        gp_total_time += dt

    gp_radii = [r["R"] for r in gp_results]
    gp_times = [r["time"] for r in gp_results]
    gp_preds = [r["pred"] for r in gp_results]
    gp_correct = sum(1 for r in gp_results if r["pred"] == r["label"]) / len(gp_results)

    print(f"  Avg time/obs: {np.mean(gp_times)*1000:.2f} ms")
    print(f"  Avg radius:   {np.mean(gp_radii):.4f}")
    print(f"  Median radius: {np.median(gp_radii):.4f}")
    print(f"  Accuracy:     {gp_correct:.3f}")
    print(f"  Total time:   {gp_total_time:.2f} s")
    print()

    # --- Head-to-head comparison ---
    print("=== Head-to-Head Comparison ===")
    speedup = np.mean(rs_times) / np.mean(gp_times) if np.mean(gp_times) > 0 else float("inf")
    print(f"  GP speedup over RS:  {speedup:.1f}x")
    print(f"  RS avg time: {np.mean(rs_times)*1000:.2f} ms  |  GP avg time: {np.mean(gp_times)*1000:.2f} ms")
    print(f"  RS avg R:    {np.mean(rs_radii):.4f}     |  GP avg R:    {np.mean(gp_radii):.4f}")
    print(f"  RS accuracy: {rs_correct:.3f}       |  GP accuracy: {gp_correct:.3f}")

    # Per-observation radius comparison
    r_diff = np.array(gp_radii) - np.array(rs_radii)
    n_gp_better = np.sum(r_diff > 0.001)
    n_rs_better = np.sum(r_diff < -0.001)
    n_tied = len(r_diff) - n_gp_better - n_rs_better
    print(f"\n  GP radius > RS: {n_gp_better}/{len(idx)}")
    print(f"  RS radius > GP: {n_rs_better}/{len(idx)}")
    print(f"  ~Tied (±0.001): {n_tied}/{len(idx)}")

    # Prediction agreement
    agree = sum(1 for rp, gp in zip(rs_preds, gp_preds) if rp == gp)
    print(f"  Prediction agreement: {agree}/{len(idx)} ({agree/len(idx):.1%})")

    # Breakdown by label
    for label, name in [(0, "clean"), (1, "adversarial")]:
        rs_r = [r["R"] for r in rs_results if r["label"] == label]
        gp_r = [r["R"] for r in gp_results if r["label"] == label]
        rs_t = [r["time"] for r in rs_results if r["label"] == label]
        gp_t = [r["time"] for r in gp_results if r["label"] == label]
        if rs_r:
            print(f"\n  [{name} obs] RS avg_R={np.mean(rs_r):.4f}  "
                  f"GP avg_R={np.mean(gp_r):.4f}  "
                  f"RS time={np.mean(rs_t)*1000:.2f}ms  "
                  f"GP time={np.mean(gp_t)*1000:.2f}ms")

    # Save results
    if args.output_json:
        import json
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        results = {
            "rs": {
                "avg_time_ms": float(np.mean(rs_times) * 1000),
                "avg_radius": float(np.mean(rs_radii)),
                "median_radius": float(np.median(rs_radii)),
                "accuracy": float(rs_correct),
                "n_samples": args.n_samples,
            },
            "gp": {
                "avg_time_ms": float(np.mean(gp_times) * 1000),
                "avg_radius": float(np.mean(gp_radii)),
                "median_radius": float(np.median(gp_radii)),
                "accuracy": float(gp_correct),
            },
            "speedup": float(speedup),
            "sigma": args.sigma,
            "n_obs": len(idx),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
