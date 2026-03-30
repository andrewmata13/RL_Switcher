import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitcherMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


class SwitcherDeepMLP(nn.Module):
    """Multi-layer binary switcher for RS certification.

    Deeper model → higher accuracy but slower per-sample forward pass,
    which directly increases RS certification time (n_samples × forward_cost).
    """
    def __init__(self, obs_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SwitcherRobustMLP(nn.Module):
    """Wide, deep binary switcher with BatchNorm and Dropout for RS certification.

    Larger capacity and BN regularization produce higher accuracy and more
    confident predictions, which translate directly to larger certified radii
    (R = sigma * Phi^-1(p_A_lower)).  BN uses running stats at eval time so
    RS certification (model.eval() + batched noisy samples) is correct.
    Dropout is disabled at eval time — no stochasticity during certification.

    Default arch: [1024, 1024, 512, 512, 256] — much wider than SwitcherDeepMLP.
    """
    def __init__(self, obs_dim: int, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024, 512, 512, 256]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_switcher(ckpt: dict):
    """Reconstruct any switcher model from a saved checkpoint dict."""
    obs_dim = int(ckpt["obs_dim"])
    model_type = ckpt.get("model_type", "mlp")
    if model_type == "robust":
        model = SwitcherRobustMLP(obs_dim=obs_dim,
                                   hidden_dims=ckpt["hidden_dims"],
                                   dropout=float(ckpt.get("dropout", 0.0)))
    elif "hidden_dims" in ckpt:
        model = SwitcherDeepMLP(obs_dim=obs_dim, hidden_dims=ckpt["hidden_dims"])
    else:
        model = SwitcherMLP(obs_dim=obs_dim, hidden_dim=int(ckpt["hidden_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
