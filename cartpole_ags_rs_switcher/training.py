from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import SwitcherMLP
from .utils import normalize


def train_switcher(
    X: np.ndarray,
    y: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 128,
    sigma: float = 0.0,
    device: Optional[torch.device] = None,
) -> SwitcherMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store raw X as tensor for noise augmentation; normalize after adding noise
    X_raw = torch.tensor(X.astype(np.float32))
    mean_t = torch.tensor(state_mean.astype(np.float32))
    std_t = torch.tensor(state_std.astype(np.float32))
    ds = TensorDataset(X_raw, torch.tensor(y.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = SwitcherMLP(obs_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - int(y.sum()), 1)
    pos_weight = torch.tensor([neg / pos], device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = (xb - mean_t.to(device)) / std_t.to(device)
            if sigma > 0.0:
                xb = xb + sigma * torch.randn_like(xb)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"[switcher] epoch={epoch+1:03d} loss={total_loss/len(ds):.4f}")

    return model.cpu()
