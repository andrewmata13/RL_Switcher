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
    """

    def __init__(
        self,
        model: SwitcherMLP,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        sigma: float,
        n_samples: int = 1000,
        confidence: float = 0.001,
    ):
        self.model = model.eval()
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.sigma = sigma
        self.n_samples = n_samples
        self.confidence = confidence  # one-sided Clopper-Pearson level alpha

    def predict(self, obs: np.ndarray) -> int:
        """Single deterministic forward pass (no smoothing). 1=critical, 0=non-critical."""
        x_norm = normalize(obs, self.state_mean, self.state_std).astype(np.float32)
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = self.model(x_t).item()
        return 1 if logit > 0 else 0

    def certify(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Monte Carlo RS certification in normalized observation space.

        Returns (prediction, p_A_lower, R):
            prediction : 0=non-critical, 1=critical, -1=abstain
            p_A_lower  : Clopper-Pearson lower bound on P(majority class)
            R          : certified L2 radius in normalized obs space (0.0 if abstaining)
        """
        obs_norm = normalize(obs, self.state_mean, self.state_std).astype(np.float32)
        noise = np.random.randn(self.n_samples, obs_norm.shape[0]).astype(np.float32) * self.sigma
        noisy_norm = obs_norm[None, :] + noise

        x_t = torch.tensor(noisy_norm, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_t)

        n_critical = int((logits > 0).sum().item())
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
