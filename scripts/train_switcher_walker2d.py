"""
Train the binary switcher MLP for Walker2D.

Usage:
    python3.8 scripts/train_switcher_walker2d.py \
        --dataset  data/walker2d_critical_dataset.npz \
        --output   models/walker2d_switcher.pt \
        --hidden-dim 64 --epochs 500 --sigma 0.1
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from hopper_ags_rs_switcher.training import train_switcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str,   required=True)
    parser.add_argument("--output",     type=str,   default="models/walker2d_switcher.pt")
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--sigma",      type=float, default=0.1,
                        help="Gaussian noise std (normalized space) for RS augmentation.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    data      = np.load(args.dataset)
    X, y      = data["X"], data["y"]
    mean, std = data["state_mean"], data["state_std"]

    print(f"Dataset: {len(X)} samples, critical fraction = {y.mean():.3f}")
    print(f"Training switcher: obs_dim={X.shape[1]}, hidden_dim={args.hidden_dim}, "
          f"epochs={args.epochs}, sigma={args.sigma}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = train_switcher(
        X, y, mean, std,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        sigma=args.sigma,
        device=device,
    )

    X_norm = torch.tensor((X - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_norm)
    preds = (logits > 0).long().numpy()
    acc   = (preds == y).mean()
    print(f"\nTrain accuracy: {acc:.3f}  "
          f"(baseline majority = {max(y.mean(), 1-y.mean()):.3f})")

    torch.save({"state_dict": model.state_dict(),
                "obs_dim":    X.shape[1],
                "hidden_dim": args.hidden_dim},
               args.output)
    print(f"Saved switcher to {args.output}")


if __name__ == "__main__":
    main()
