"""
Quadratic activation model and Gil-Pelaez certifier for the binary switcher.

SwitcherQuadMLP: 2-class model with x²+x activation (compatible with GP certification).
GPSwitcher: Drop-in replacement for VanillaRSSwitcher using single-pass Gil-Pelaez
            certification instead of Monte Carlo randomized smoothing.
"""
import sys
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Add Single_Pass_Smoothing to path so we can import its modules
_SPS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "Single_Pass_Smoothing")
if _SPS_ROOT not in sys.path:
    sys.path.insert(0, _SPS_ROOT)

from certification import certify_quad_pA, certified_radius  # noqa: E402
from model import QuadLinear  # noqa: E402
from moment_propagation import propagate_network  # noqa: E402


class SwitcherQuadMLP(nn.Module):
    """
    2-class binary switcher with quadratic activation.

    Architecture: Linear(obs_dim, hidden_dim) -> x²+x -> Linear(hidden_dim, 2)

    Output: 2 logits [z_noncritical, z_critical].
    pred = argmax(z): 0 = non-critical (use PPO), 1 = critical (use ATLA).

    This architecture enables exact Gil-Pelaez certification because:
    - The backbone (first Linear) is affine
    - x²+x produces a generalized chi-squared distribution under Gaussian noise
    - The margin z_0 - z_1 has a known characteristic function
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            QuadLinear(),
            nn.Linear(hidden_dim, 2),
        )
        # Xavier init for better training with x²+x
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwitcherQuadDeepMLP(nn.Module):
    """
    Deep 2-class binary switcher with stacked Linear+BN backbone and x²+x activation.

    Architecture:
        [Linear+BN] × N  →  x²+x  →  Linear(hidden, 2)

    At eval time, all Linear+BN layers fold into a single affine map, so
    certification sees Linear → x²+x → Linear (same as SwitcherQuadMLP).
    During training, BatchNorm helps optimization significantly — it keeps
    pre-activations well-scaled for x²+x and acts as a regularizer.

    This is the 1D analogue of QuadConvNet from Single_Pass_Smoothing.
    """

    def __init__(self, obs_dim: int, backbone_dims: list = None):
        super().__init__()
        if backbone_dims is None:
            backbone_dims = [128, 256, 256]

        # Stacked Linear+BN backbone (linear at eval time)
        layers = []
        in_dim = obs_dim
        for h in backbone_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.quad = QuadLinear()
        self.classifier = nn.Linear(backbone_dims[-1], 2)

        # Init
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.quad(h)
        return self.classifier(h)

    def fold_backbone(self) -> nn.Sequential:
        """Fold all Linear+BN layers into a single Linear for GP certification.

        Returns nn.Sequential(Linear, QuadLinear, Linear) — the same structure
        as SwitcherQuadMLP.net, compatible with certify_quad_pA and
        propagate_network.
        """
        self.eval()
        W_fused = None
        b_fused = None

        for module in self.backbone:
            if isinstance(module, nn.Linear):
                W = module.weight.detach()
                b = module.bias.detach()
                if W_fused is None:
                    W_fused = W
                    b_fused = b
                else:
                    b_fused = W @ b_fused + b
                    W_fused = W @ W_fused

            elif isinstance(module, nn.BatchNorm1d):
                # BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
                scale = module.weight.detach() / torch.sqrt(
                    module.running_var + module.eps)
                shift = module.bias.detach() - scale * module.running_mean
                # Apply to current fused: x -> scale * (W_fused @ input + b_fused) + shift
                W_fused = scale.unsqueeze(1) * W_fused
                b_fused = scale * b_fused + shift

        folded_linear = nn.Linear(W_fused.shape[1], W_fused.shape[0], bias=True)
        folded_linear.weight = nn.Parameter(W_fused)
        folded_linear.bias = nn.Parameter(b_fused)

        return nn.Sequential(folded_linear, QuadLinear(), self.classifier)


class GPSwitcher:
    """
    Gil-Pelaez single-pass certifier for the binary switcher.

    Drop-in replacement for VanillaRSSwitcher: same certify() API returning
    (pred, p_A_lower, R), but uses analytical Gil-Pelaez inversion instead of
    Monte Carlo sampling.

    For binary classification (2 classes), there is NO union bound penalty —
    the Gil-Pelaez certificate is exact.

    For 11-dim Hopper obs: eigendecomposition is O(11³) = trivial.
    Expected certification time: <1ms per observation (vs ~2-50ms for MC RS
    with 10k samples on GPU).
    """

    def __init__(
        self,
        model,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        sigma: float,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device)
        self.state_std = torch.tensor(state_std, dtype=torch.float32, device=self.device)
        self.sigma = sigma

        # For deep models, fold backbone into single Linear for certification
        if isinstance(model, SwitcherQuadDeepMLP):
            self._cert_net = model.fold_backbone().to(self.device)
        else:
            self._cert_net = model.net

    def _normalize_t(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return (x - self.state_mean) / (self.state_std + 1e-8)

    def predict(self, obs: np.ndarray) -> int:
        """Single deterministic forward pass (no smoothing). 1=critical, 0=non-critical."""
        x_norm = self._normalize_t(obs).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x_norm).squeeze(0)
        return int(logits.argmax().item())

    def certify(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Exact Gil-Pelaez certification in normalized observation space.

        Returns (prediction, p_A_lower, R):
            prediction : 0=non-critical, 1=critical
            p_A_lower  : exact lower bound on P(predicted class wins under noise)
            R          : certified L2 radius in normalized obs space
        """
        x_norm = self._normalize_t(obs)

        # Clean forward pass to get predicted class (use folded net)
        with torch.no_grad():
            logits = self._cert_net(x_norm.unsqueeze(0)).squeeze(0)
        pred = int(logits.argmax().item())

        # Moment propagation through the folded network
        obs_dim = x_norm.shape[0]
        Sigma_in = (self.sigma ** 2) * torch.eye(obs_dim, device=self.device)
        mu_out, _ = propagate_network(self._cert_net, x_norm, Sigma_in, K=5)

        # Gil-Pelaez exact certification
        pA = certify_quad_pA(
            self._cert_net, x_norm, mu_out, self.sigma,
            predicted_class=pred,
        )

        R = certified_radius(pA, self.sigma)
        return pred, float(pA), float(R)

    def certify_timed(self, obs: np.ndarray) -> Tuple[int, float, float, float]:
        """Like certify(), but also returns wall-clock time in seconds."""
        t0 = time.perf_counter()
        pred, pA, R = self.certify(obs)
        dt = time.perf_counter() - t0
        return pred, pA, R, dt
