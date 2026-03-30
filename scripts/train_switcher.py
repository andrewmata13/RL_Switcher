"""
Train a binary switcher for RS certification.

Supports three model types:
  mlp    — SwitcherMLP: single hidden layer (fast, low capacity)
  deep   — SwitcherDeepMLP: multi-layer ReLU, no BN/dropout
  robust — SwitcherRobustMLP: wide layers + BatchNorm + Dropout (best certified radii)

Usage (robust, recommended for MuJoCo):
    python3.8 scripts/train_switcher.py \
        --dataset data/hopper_critical_dataset.npz \
        --output models/hopper_switcher_robust.pt \
        --model-type robust \
        --hidden-dims 1024,1024,512,512,256 \
        --dropout 0.1 \
        --epochs 500 \
        --sigma 0.1 \
        --n-noise-copies 4 \
        --batch-size 256 \
        --lr 3e-4 \
        --weight-decay 1e-4
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from rs_switcher_common.training import train_switcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str,   required=True)
    parser.add_argument("--output",       type=str,   default="models/switcher.pt")
    parser.add_argument("--model-type",   type=str,   default="robust",
                        choices=["mlp", "deep", "robust"])
    parser.add_argument("--hidden-dim",   type=int,   default=64,
                        help="Hidden dim for mlp model type")
    parser.add_argument("--hidden-dims",  type=str,   default=None,
                        help="Comma-separated layer widths, e.g. 1024,1024,512,512,256")
    parser.add_argument("--dropout",      type=float, default=0.1,
                        help="Dropout rate (robust model only; 0=disabled)")
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--sigma",        type=float, default=0.1,
                        help="Gaussian noise std for RS training augmentation")
    parser.add_argument("--n-noise-copies", type=int, default=4,
                        help="Noise augmentation copies per sample per step")
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule",  type=str,   default="cosine",
                        choices=["cosine", "none"])
    args = parser.parse_args()

    hidden_dims = None
    if args.hidden_dims is not None:
        hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    elif args.model_type in ("deep", "robust"):
        hidden_dims = [1024, 1024, 512, 512, 256]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    mean, std = data["state_mean"], data["state_std"]

    print(f"Dataset: {len(X)} samples, obs_dim={X.shape[1]}, "
          f"pos={int(y.sum())}, neg={len(y)-int(y.sum())}")
    print(f"Model: {args.model_type}  hidden_dims={hidden_dims}  dropout={args.dropout}")
    print(f"Training: epochs={args.epochs}  sigma={args.sigma}  "
          f"n_noise_copies={args.n_noise_copies}  lr={args.lr}  "
          f"weight_decay={args.weight_decay}  lr_schedule={args.lr_schedule}")
    print()

    model = train_switcher(
        X, y, mean, std,
        hidden_dim=args.hidden_dim,
        hidden_dims=hidden_dims,
        model_type=args.model_type,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        sigma=args.sigma,
        n_noise_copies=args.n_noise_copies,
        lr_schedule=args.lr_schedule,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Evaluate accuracy on clean data
    model.eval()
    X_norm = (X - mean) / (std + 1e-8)
    with torch.no_grad():
        logits = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
    preds = (logits > 0).astype(int)
    acc = (preds == y.astype(int)).mean()
    print(f"\nTrain accuracy (clean): {acc*100:.1f}%")

    ckpt = {
        "state_dict": model.state_dict(),
        "obs_dim":    X.shape[1],
        "model_type": args.model_type,
        "hidden_dims": hidden_dims,
        "hidden_dim":  args.hidden_dim,
        "dropout":     args.dropout,
        "sigma":       args.sigma,
    }
    torch.save(ckpt, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
