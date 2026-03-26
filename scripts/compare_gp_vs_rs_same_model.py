"""
Fair GP vs RS comparison on the SAME SwitcherQuadMLP model.

Both methods certify the same observations using the same network, so
any difference in certified radius is due to the certification method,
not the model. RS radii should converge to GP radii as n_samples → ∞
(GP is exact; RS is a statistical lower bound).

Also compares wall-clock time at multiple n_samples values.

Usage:
    python3.8 scripts/compare_gp_vs_rs_same_model.py \
        --dataset data/hopper_critical_dataset.npz \
        --gp-switcher-path models/hopper_switcher_gp.pt \
        --sigma 0.1 --n-obs 200
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import json
import numpy as np
import torch
from scipy.stats import beta, norm

from rs_switcher_common.gp_models import SwitcherQuadMLP, SwitcherQuadDeepMLP, GPSwitcher


def rs_certify_2class(model, obs_norm_t, sigma, n_samples, confidence=0.001):
    """MC RS certification for a 2-class model (argmax instead of threshold)."""
    device = obs_norm_t.device
    noise = torch.randn(n_samples, obs_norm_t.shape[0], device=device) * sigma
    noisy = obs_norm_t.unsqueeze(0) + noise

    with torch.no_grad():
        logits = model(noisy)  # (n_samples, 2)

    preds = logits.argmax(dim=1)  # (n_samples,)
    n_class0 = int((preds == 0).sum().item())
    n_class1 = n_samples - n_class0

    if n_class0 >= n_class1:
        cA, nA, nB = 0, n_class0, n_class1
    else:
        cA, nA, nB = 1, n_class1, n_class0

    p_A_lower = float(beta.ppf(confidence, nA, nB + 1))

    if p_A_lower <= 0.5:
        return -1, p_A_lower, 0.0

    R = float(sigma * norm.ppf(p_A_lower))
    return cA, p_A_lower, R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gp-switcher-path", type=str, required=True)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--n-obs", type=int, default=200)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    mean, std = data["state_mean"], data["state_std"]

    # Load the single shared model
    gp_ck = torch.load(args.gp_switcher_path, map_location="cpu")
    if gp_ck.get("model_type") == "quad_deep":
        model = SwitcherQuadDeepMLP(obs_dim=int(gp_ck["obs_dim"]),
                                     backbone_dims=gp_ck["backbone_dims"])
        model.load_state_dict(gp_ck["state_dict"])
        model.eval()
        arch_str = f"SwitcherQuadDeepMLP, backbone={gp_ck['backbone_dims']}"
    else:
        model = SwitcherQuadMLP(obs_dim=int(gp_ck["obs_dim"]),
                                 hidden_dim=int(gp_ck["hidden_dim"]))
        model.load_state_dict(gp_ck["state_dict"])
        model.eval()
        arch_str = f"SwitcherQuadMLP, hidden_dim={gp_ck['hidden_dim']}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch_str}, params={n_params:,}")
    print(f"Dataset: {len(X)} samples, obs_dim={X.shape[1]}")

    gp_cert = GPSwitcher(model, mean, std, sigma=args.sigma, device="cpu")

    # Select balanced observations
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

    # Normalize observations once
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    obs_norms = []
    for ix in idx:
        x = torch.tensor(X[ix], dtype=torch.float32)
        obs_norms.append((x - mean_t) / (std_t + 1e-8))

    # --- GP certification ---
    print(f"\n=== GP Certification (exact, single-pass) ===")
    gp_radii = []
    gp_preds = []
    gp_pAs = []
    gp_times = []
    for i, ix in enumerate(idx):
        pred, pA, R, dt = gp_cert.certify_timed(X[ix])
        gp_radii.append(R)
        gp_preds.append(pred)
        gp_pAs.append(pA)
        gp_times.append(dt)

    gp_correct = sum(1 for i, ix in enumerate(idx)
                     if gp_preds[i] == y[ix]) / len(idx)
    print(f"  Avg radius:   {np.mean(gp_radii):.4f}")
    print(f"  Median radius: {np.median(gp_radii):.4f}")
    print(f"  Avg pA:       {np.mean(gp_pAs):.4f}")
    print(f"  Accuracy:     {gp_correct:.3f}")
    print(f"  Avg time:     {np.mean(gp_times)*1000:.2f} ms")

    # --- RS certification at multiple sample counts ---
    n_samples_list = [1000, 10000, 50000, 100000]
    all_rs = {}

    for n_samples in n_samples_list:
        print(f"\n=== RS Certification (n_samples={n_samples:,}, CPU) ===")
        rs_radii = []
        rs_preds = []
        rs_pAs = []
        rs_times = []
        for i, obs_norm in enumerate(obs_norms):
            t0 = time.perf_counter()
            pred, pA, R = rs_certify_2class(model, obs_norm, args.sigma,
                                             n_samples, confidence=0.001)
            dt = time.perf_counter() - t0
            rs_radii.append(R)
            rs_preds.append(pred)
            rs_pAs.append(pA)
            rs_times.append(dt)

        rs_correct = sum(1 for i, ix in enumerate(idx)
                         if rs_preds[i] == y[ix]) / len(idx)
        print(f"  Avg radius:   {np.mean(rs_radii):.4f}")
        print(f"  Median radius: {np.median(rs_radii):.4f}")
        print(f"  Avg pA:       {np.mean(rs_pAs):.4f}")
        print(f"  Accuracy:     {rs_correct:.3f}")
        print(f"  Avg time:     {np.mean(rs_times)*1000:.2f} ms")

        # Per-obs radius comparison with GP
        r_diff = np.array(gp_radii) - np.array(rs_radii)
        # Only compare where both methods agree on prediction
        agree_mask = [gp_preds[i] == rs_preds[i] for i in range(len(idx))]
        n_agree = sum(agree_mask)
        if n_agree > 0:
            agree_diff = [r_diff[i] for i in range(len(idx)) if agree_mask[i]]
            print(f"  Pred agreement: {n_agree}/{len(idx)} ({n_agree/len(idx):.1%})")
            print(f"  GP−RS radius (where agreed): "
                  f"mean={np.mean(agree_diff):.4f}, "
                  f"median={np.median(agree_diff):.4f}")

        speedup = np.mean(rs_times) / np.mean(gp_times)
        print(f"  GP speedup:   {speedup:.1f}x")
        fits_8ms = np.mean(rs_times)*1000 < 8.0
        print(f"  Fits in 8ms:  RS={'YES' if fits_8ms else 'NO'}, GP=YES")

        all_rs[n_samples] = {
            "avg_radius": float(np.mean(rs_radii)),
            "median_radius": float(np.median(rs_radii)),
            "avg_pA": float(np.mean(rs_pAs)),
            "accuracy": float(rs_correct),
            "avg_time_ms": float(np.mean(rs_times) * 1000),
            "speedup": float(speedup),
        }

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"SUMMARY: Same model ({n_params:,} params), sigma={args.sigma}")
    print(f"{'='*70}")
    print(f"{'Method':<16} {'Avg R':>8} {'Med R':>8} {'Avg pA':>8} "
          f"{'Time(ms)':>10} {'Fits 8ms':>9}")
    print(f"{'-'*70}")
    print(f"{'GP (exact)':<16} {np.mean(gp_radii):>8.4f} "
          f"{np.median(gp_radii):>8.4f} {np.mean(gp_pAs):>8.4f} "
          f"{np.mean(gp_times)*1000:>10.2f} {'YES':>9}")
    for ns in n_samples_list:
        r = all_rs[ns]
        fits = "YES" if r["avg_time_ms"] < 8.0 else "NO"
        print(f"{'RS@'+str(ns//1000)+'k':<16} {r['avg_radius']:>8.4f} "
              f"{r['median_radius']:>8.4f} {r['avg_pA']:>8.4f} "
              f"{r['avg_time_ms']:>10.2f} {fits:>9}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        results = {
            "model": {
                "type": "SwitcherQuadMLP",
                "hidden_dim": int(gp_ck["hidden_dim"]),
                "params": n_params,
            },
            "sigma": args.sigma,
            "n_obs": len(idx),
            "gp": {
                "avg_radius": float(np.mean(gp_radii)),
                "median_radius": float(np.median(gp_radii)),
                "avg_pA": float(np.mean(gp_pAs)),
                "accuracy": float(gp_correct),
                "avg_time_ms": float(np.mean(gp_times) * 1000),
            },
            "rs": all_rs,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
