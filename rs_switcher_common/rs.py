from typing import Tuple

import numpy as np
import torch
from scipy.stats import beta, norm

from .models import SwitcherMLP
from .utils import normalize


class VanillaRSSwitcher:
    """
    Randomized smoothing certification for the binary switcher via Monte Carlo sampling.

    Adds isotropic Gaussian noise (std=sigma) in NORMALIZED observation space, runs the
    switcher on each noisy copy, and derives a Clopper-Pearson lower bound on the
    majority-class probability. The certified L2 radius is R = sigma * Phi^{-1}(p_A_lower),
    measured in normalized observation space.

    GPU support: pass device="cuda" to run noise sampling and inference on GPU.
    With n_samples=10000 on GPU, certify() runs in ~2ms vs ~50ms on CPU.
    """

    def __init__(
        self,
        model: SwitcherMLP,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        sigma: float,
        n_samples: int = 1000,
        confidence: float = 0.001,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device)
        self.state_std  = torch.tensor(state_std,  dtype=torch.float32, device=self.device)
        self.sigma = sigma
        self.n_samples = n_samples
        self.confidence = confidence  # one-sided Clopper-Pearson level alpha

    def _normalize_t(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return (x - self.state_mean) / (self.state_std + 1e-8)

    def predict(self, obs: np.ndarray) -> int:
        """Single deterministic forward pass (no smoothing). 1=critical, 0=non-critical."""
        x_norm = self._normalize_t(obs).unsqueeze(0)
        with torch.no_grad():
            logit = self.model(x_norm).item()
        return 1 if logit > 0 else 0

    def certify(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Monte Carlo RS certification in normalized observation space.

        Noise sampling and inference run on self.device (GPU if available).

        Returns (prediction, p_A_lower, R):
            prediction : 0=non-critical, 1=critical, -1=abstain
            p_A_lower  : Clopper-Pearson lower bound on P(majority class)
            R          : certified L2 radius in normalized obs space (0.0 if abstaining)
        """
        obs_norm = self._normalize_t(obs)                          # (obs_dim,)
        noise    = torch.randn(self.n_samples, obs_norm.shape[0],
                               device=self.device) * self.sigma    # (n, obs_dim)
        noisy    = obs_norm.unsqueeze(0) + noise                   # (n, obs_dim)

        with torch.no_grad():
            logits = self.model(noisy)                             # (n, 1) or (n,)

        n_critical    = int((logits > 0).sum().item())
        n_noncritical = self.n_samples - n_critical

        if n_noncritical >= n_critical:
            cA, nA, nB = 0, n_noncritical, n_critical
        else:
            cA, nA, nB = 1, n_critical, n_noncritical

        # One-sided Clopper-Pearson lower bound at level self.confidence
        p_A_lower = float(beta.ppf(self.confidence, nA, nB + 1))

        if p_A_lower <= 0.5:
            return -1, p_A_lower, 0.0

        R = float(self.sigma * norm.ppf(p_A_lower))
        return cA, p_A_lower, R
