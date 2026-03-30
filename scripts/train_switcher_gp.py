"""
Train a quadratic-activation switcher for Gil-Pelaez certification.

Supports two architectures:
  - SwitcherQuadMLP:     Linear → x²+x → Linear  (--hidden-dim)
  - SwitcherQuadDeepMLP: [Linear+BN]×N → x²+x → Linear  (--backbone-dims)

The deep variant stacks Linear+BN layers before x²+x. At eval time these
fold into a single affine map, so certification still sees Linear → x²+x → Linear.
During training, BatchNorm dramatically helps optimization.

Usage:
    # Shallow (original)
    python3.8 scripts/train_switcher_gp.py \
        --dataset data/hopper_critical_dataset.npz \
        --output  models/hopper_switcher_gp.pt \
        --hidden-dim 64 --epochs 500 --sigma 0.1

    # Deep (with BN backbone)
    python3.8 scripts/train_switcher_gp.py \
        --dataset data/hopper_critical_dataset.npz \
        --output  models/hopper_switcher_gp_deep.pt \
        --backbone-dims 128 256 256 --epochs 500 --sigma 0.1
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rs_switcher_common.gp_models import (
    SwitcherQuadMLP, SwitcherQuadDeepMLP, SwitcherQuadSkipMLP,
    SwitcherBottleneckMLP,
)


def margin_loss(logits, labels, kappa=2.0):
    """Hinge-style margin loss: encourage correct-class margin >= kappa."""
    batch_size = logits.shape[0]
    correct_logits = logits[torch.arange(batch_size), labels]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(batch_size), labels] = False
    wrong_logits = logits.masked_fill(~mask, float('-inf')).max(dim=1).values
    margin = correct_logits - wrong_logits
    return torch.clamp(kappa - margin, min=0.0).mean()


def train_gp_switcher(
    X: np.ndarray,
    y: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    hidden_dim: int = 64,
    backbone_dims: list = None,
    skip_dim: int = 0,
    arch: str = "quad",
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 128,
    sigma: float = 0.1,
    lambda_margin: float = 0.5,
    kappa: float = 2.0,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize data
    mean_t = torch.tensor(state_mean.astype(np.float32), device=device)
    std_t = torch.tensor(state_std.astype(np.float32), device=device)

    X_raw = torch.tensor(X.astype(np.float32))
    y_long = torch.tensor(y.astype(np.int64))
    ds = TensorDataset(X_raw, y_long)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if arch == "bottleneck":
        model = SwitcherBottleneckMLP(obs_dim=X.shape[1],
                                      hidden_dim=hidden_dim).to(device)
    elif skip_dim > 0:
        model = SwitcherQuadSkipMLP(obs_dim=X.shape[1],
                                     quad_dim=hidden_dim,
                                     skip_dim=skip_dim).to(device)
    elif backbone_dims is not None:
        model = SwitcherQuadDeepMLP(obs_dim=X.shape[1],
                                     backbone_dims=backbone_dims).to(device)
    else:
        model = SwitcherQuadMLP(obs_dim=X.shape[1],
                                 hidden_dim=hidden_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Pos weight for class imbalance (if any)
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - int(y.sum()), 1)
    class_weights = torch.tensor([1.0, neg / pos], device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            # Normalize
            xb = (xb - mean_t) / std_t

            # Noise augmentation in normalized space
            if sigma > 0.0:
                xb = xb + sigma * torch.randn_like(xb)

            logits = model(xb)

            # CE + margin loss
            loss_ce = F.cross_entropy(logits, yb, weight=class_weights)
            loss_margin = margin_loss(logits, yb, kappa=kappa)
            loss = loss_ce + lambda_margin * loss_margin

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(xb)

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            acc = correct / total
            avg_loss = total_loss / total
            print(f"[gp_switcher] epoch={epoch+1:03d} loss={avg_loss:.4f} acc={acc:.4f}")

    return model.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, required=True)
    parser.add_argument("--output",        type=str, default=None)
    parser.add_argument("--hidden-dim",    type=int, default=64,
                        help="Hidden dim for shallow SwitcherQuadMLP (ignored if --backbone-dims)")
    parser.add_argument("--backbone-dims", type=int, nargs="+", default=None,
                        help="Stacked Linear+BN dims for deep model (e.g. 128 256 256)")
    parser.add_argument("--skip-dim",     type=int, default=0,
                        help="Linear skip pathway dimension (0=disabled). "
                             "Uses SwitcherQuadSkipMLP: quad_dim=hidden-dim, skip_dim=this.")
    parser.add_argument("--arch",         type=str, default="quad",
                        choices=["quad", "bottleneck"],
                        help="Architecture: 'quad'=x²+x (default), "
                             "'bottleneck'=Linear(obs,k)->ReLU->Linear(k,2) "
                             "certified via k-dim Gauss-Hermite quadrature")
    parser.add_argument("--epochs",        type=int, default=500)
    parser.add_argument("--sigma",         type=float, default=0.1)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--lambda-margin", type=float, default=0.5)
    parser.add_argument("--kappa",         type=float, default=2.0)
    args = parser.parse_args()

    output = args.output or args.dataset.replace("_dataset.npz", "_gp_switcher.pt")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    mean, std = data["state_mean"], data["state_std"]

    print(f"Dataset: {len(X)} samples, obs_dim={X.shape[1]}, "
          f"critical fraction = {y.mean():.3f}")
    if args.arch == "bottleneck":
        print(f"Training bottleneck GP switcher: k={args.hidden_dim}, "
              f"epochs={args.epochs}, sigma={args.sigma}, "
              f"lambda_margin={args.lambda_margin}, kappa={args.kappa}")
    elif args.skip_dim > 0:
        print(f"Training quad+skip GP switcher: quad_dim={args.hidden_dim}, "
              f"skip_dim={args.skip_dim}, epochs={args.epochs}, sigma={args.sigma}, "
              f"lambda_margin={args.lambda_margin}, kappa={args.kappa}")
    elif args.backbone_dims:
        print(f"Training deep GP switcher: backbone_dims={args.backbone_dims}, "
              f"epochs={args.epochs}, sigma={args.sigma}, "
              f"lambda_margin={args.lambda_margin}, kappa={args.kappa}")
    else:
        print(f"Training GP switcher: hidden_dim={args.hidden_dim}, "
              f"epochs={args.epochs}, sigma={args.sigma}, "
              f"lambda_margin={args.lambda_margin}, kappa={args.kappa}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_gp_switcher(
        X, y, mean, std,
        hidden_dim=args.hidden_dim,
        backbone_dims=args.backbone_dims,
        skip_dim=args.skip_dim,
        arch=args.arch,
        epochs=args.epochs,
        sigma=args.sigma,
        lr=args.lr,
        lambda_margin=args.lambda_margin,
        kappa=args.kappa,
        device=device,
    )

    # Accuracy check
    model.eval()
    X_norm = torch.tensor((X - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_norm)
    preds = logits.argmax(dim=1).numpy()
    acc = (preds == y).mean()
    print(f"\nTrain accuracy: {acc:.3f}  "
          f"(baseline majority = {max(y.mean(), 1-y.mean()):.3f})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    ck = {"state_dict": model.state_dict(), "obs_dim": X.shape[1]}
    if args.arch == "bottleneck":
        ck["model_type"] = "bottleneck"
        ck["hidden_dim"] = args.hidden_dim
    elif args.skip_dim > 0:
        ck["model_type"] = "quad_skip"
        ck["quad_dim"] = args.hidden_dim
        ck["skip_dim"] = args.skip_dim
    elif args.backbone_dims:
        ck["backbone_dims"] = args.backbone_dims
        ck["model_type"] = "quad_deep"
    else:
        ck["hidden_dim"] = args.hidden_dim
        ck["model_type"] = "quad"
    torch.save(ck, output)
    print(f"Saved GP switcher to {output}")


if __name__ == "__main__":
    main()
