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
