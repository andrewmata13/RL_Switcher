"""
Train the binary switcher MLP for MuJoCo environments.

Usage:
    python3.8 scripts/train_switcher_mujoco.py --env hopper \
        --dataset  data/hopper_critical_dataset.npz \
        --output   models/hopper_switcher.pt \
        --hidden-dim 64 --epochs 500 --sigma 0.1
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from rs_switcher_common.training import train_switcher
from rs_switcher_common.env_config import ENV_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None,
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment name (used for default output path)")
    parser.add_argument("--dataset",    type=str,   required=True)
    parser.add_argument("--output",     type=str,   default=None,
                        help="Output path (default: models/{env}_switcher.pt)")
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--hidden-dims", type=int,  nargs="+", default=None,
                        help="Multi-layer hidden dims (e.g. 256 128 64). Overrides --hidden-dim.")
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--sigma",      type=float, default=0.1,
                        help="Gaussian noise std (normalized space) for RS augmentation.")
    args = parser.parse_args()

    output = args.output
    if output is None:
        env_name = args.env or "mujoco"
        output = f"models/{env_name}_switcher.pt"

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    mean, std = data["state_mean"], data["state_std"]

    if args.hidden_dims:
        print(f"Dataset: {len(X)} samples, obs_dim={X.shape[1]}, "
              f"critical fraction = {y.mean():.3f}")
        print(f"Training deep switcher: hidden_dims={args.hidden_dims}, "
              f"epochs={args.epochs}, sigma={args.sigma}")
    else:
        print(f"Dataset: {len(X)} samples, obs_dim={X.shape[1]}, "
              f"critical fraction = {y.mean():.3f}")
        print(f"Training switcher: hidden_dim={args.hidden_dim}, "
              f"epochs={args.epochs}, sigma={args.sigma}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_switcher(
        X, y, mean, std,
        hidden_dim=args.hidden_dim,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        sigma=args.sigma,
        device=device,
    )

    # Quick accuracy check
    X_norm = torch.tensor((X - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_norm)
    preds = (logits > 0).long().numpy()
    acc = (preds == y).mean()
    print(f"\nTrain accuracy: {acc:.3f}  "
          f"(baseline majority = {max(y.mean(), 1-y.mean()):.3f})")

    ck = {"state_dict": model.state_dict(), "obs_dim": X.shape[1]}
    if args.hidden_dims:
        ck["hidden_dims"] = args.hidden_dims
    else:
        ck["hidden_dim"] = args.hidden_dim
    torch.save(ck, output)
    print(f"Saved switcher to {output}")


if __name__ == "__main__":
    main()
