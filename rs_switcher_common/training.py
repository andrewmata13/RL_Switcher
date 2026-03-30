from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import SwitcherMLP, SwitcherDeepMLP, SwitcherRobustMLP
from .utils import normalize


def train_switcher(
    X: np.ndarray,
    y: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    hidden_dim: int = 32,
    hidden_dims: list = None,
    model_type: str = "mlp",
    dropout: float = 0.1,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    sigma: float = 0.0,
    n_noise_copies: int = 1,
    lr_schedule: str = "cosine",
    device: Optional[torch.device] = None,
):
    """Train a binary switcher with RS noise augmentation.

    model_type:
      "mlp"    — SwitcherMLP (single hidden layer)
      "deep"   — SwitcherDeepMLP (multi-layer ReLU, hidden_dims list)
      "robust" — SwitcherRobustMLP (wide+BN+dropout, best for RS certification)

    n_noise_copies: number of independent Gaussian noise samples added per
      training example per step.  Higher values (4–8) better approximate the
      smoothed classifier's training objective and tend to increase certified
      radii.  Effective batch size becomes batch_size * n_noise_copies.

    lr_schedule: "cosine" (CosineAnnealingLR) or "none".
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_raw = torch.tensor(X.astype(np.float32))
    mean_t = torch.tensor(state_mean.astype(np.float32))
    std_t  = torch.tensor(state_std.astype(np.float32))
    ds = TensorDataset(X_raw, torch.tensor(y.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    obs_dim = X.shape[1]
    if model_type == "robust":
        model = SwitcherRobustMLP(obs_dim=obs_dim,
                                   hidden_dims=hidden_dims,
                                   dropout=dropout).to(device)
    elif model_type == "deep" or hidden_dims is not None:
        model = SwitcherDeepMLP(obs_dim=obs_dim, hidden_dims=hidden_dims).to(device)
    else:
        model = SwitcherMLP(obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    else:
        scheduler = None

    pos = max(int(y.sum()), 1)
    neg = max(len(y) - int(y.sum()), 1)
    pos_weight = torch.tensor([neg / pos], device=device)

    mean_d = mean_t.to(device)
    std_d  = std_t.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_total = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            xb_norm = (xb - mean_d) / (std_d + 1e-8)

            if sigma > 0.0 and n_noise_copies > 1:
                # Expand each sample to n_noise_copies noisy versions
                xb_exp = xb_norm.repeat_interleave(n_noise_copies, dim=0)
                yb_exp = yb.repeat_interleave(n_noise_copies, dim=0)
                xb_exp = xb_exp + sigma * torch.randn_like(xb_exp)
            else:
                xb_exp = xb_norm + sigma * torch.randn_like(xb_norm) if sigma > 0.0 else xb_norm
                yb_exp = yb

            logits = model(xb_exp)
            loss = F.binary_cross_entropy_with_logits(logits, yb_exp, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb_exp)
            n_total += len(xb_exp)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 50 == 0:
            lr_cur = opt.param_groups[0]["lr"]
            print(f"[switcher] epoch={epoch+1:04d}  loss={total_loss/n_total:.4f}  lr={lr_cur:.2e}")

    return model.cpu()
