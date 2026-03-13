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
