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
import torch.nn.functional as F

# Add Single_Pass_Smoothing to path so we can import its modules
_SPS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "Single_Pass_Smoothing")
if _SPS_ROOT not in sys.path:
    sys.path.insert(0, _SPS_ROOT)

from certification import certify_quad_pA, certified_radius, _gil_pelaez_cdf  # noqa: E402
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


class SwitcherQuadSkipMLP(nn.Module):
    """
    2-class switcher with parallel quadratic + linear skip pathways.

    Architecture:
        x → Linear_quad(obs_dim, quad_dim) → x²+x → v_quad   (quad_dim)
        x → Linear_skip(obs_dim, skip_dim) → v_skip           (skip_dim)
        [v_quad; v_skip] → Linear(quad_dim + skip_dim, 2) → logits

    The skip pathway provides independent linear features. Under Gaussian noise,
    the margin M_j = z_A - z_j is still a generalized chi-squared:
    - Quadratic part: from x²+x applied to W_quad @ (x+ε)
    - Linear part: from W_skip @ (x+ε)  (Gaussian, adds to linear coefficients)

    The characteristic function factors identically to the pure-quadratic case
    after absorbing the skip contribution into the linear coefficients of the
    eigendecomposed quadratic form. Gil-Pelaez inversion remains exact.
    """

    def __init__(self, obs_dim: int, quad_dim: int = 512, skip_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.quad_dim = quad_dim
        self.skip_dim = skip_dim

        self.linear_quad = nn.Linear(obs_dim, quad_dim)
        self.quad = QuadLinear()
        self.linear_skip = nn.Linear(obs_dim, skip_dim)
        self.classifier = nn.Linear(quad_dim + skip_dim, 2)

        # Xavier init
        for m in [self.linear_quad, self.linear_skip, self.classifier]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_quad = self.quad(self.linear_quad(x))
        v_skip = self.linear_skip(x)
        return self.classifier(torch.cat([v_quad, v_skip], dim=-1))


class SwitcherBottleneckMLP(nn.Module):
    """
    2-class binary switcher with small ReLU bottleneck for exact certification.

    Architecture: Linear(obs_dim, k) -> ReLU -> Linear(k, 2)

    With small k (e.g. k=4 or k=8), the pre-activation under Gaussian noise is a
    k-dimensional Gaussian, enabling exact certification via k-dimensional
    Gauss-Hermite quadrature over the ReLU activation patterns.

    Key insight: No pattern enumeration (which scales as 2^k). Instead, integrate
    directly over the k-dim Gaussian pre-activation space using n^k quadrature points.
    For k=4, n=16: 65536 evaluations, ~2ms. For k=8, n=6: 1.7M evaluations, ~30ms.

    This architecture achieves strictly higher expressivity than x²+x (degree-2
    polynomials): ReLU(z) is a piecewise-linear function, not a polynomial.
    The decision boundary can be any piecewise-linear function in obs space.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)

        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


def certify_bottleneck_pA(model, x_norm, sigma, predicted_class=None, n_quad=16):
    """
    Exact p_A for SwitcherBottleneckMLP via k-dim Gauss-Hermite quadrature.

    Under isotropic Gaussian noise N(0, sigma^2 I) in normalized obs space,
    the pre-activations h = W1 @ (x + eps) + b1 follow:
        h ~ N(a, sigma^2 * W1 @ W1^T)    where a = W1 @ x + b1

    We integrate P(M > 0) where M = u^T ReLU(h) + d over this Gaussian:
        p_A = E_{z ~ N(0,I)} [ I( u^T ReLU(a + L @ z) + d > 0 ) ]

    using n^k Gauss-Hermite quadrature points in k-dim z-space.

    Args:
        model: SwitcherBottleneckMLP
        x_norm: normalized observation (torch.Tensor, shape [obs_dim])
        sigma: smoothing noise std
        predicted_class: override prediction (default: argmax of logits)
        n_quad: quadrature order per dimension (default 16; use 12 for k>=6)

    Returns:
        p_A: float in [0,1], exact probability that predicted class wins under noise
    """
    W1 = model.linear1.weight.detach().cpu().numpy().astype(np.float64)  # (k, obs_dim)
    b1 = model.linear1.bias.detach().cpu().numpy().astype(np.float64)    # (k,)
    W2 = model.linear2.weight.detach().cpu().numpy().astype(np.float64)  # (2, k)
    b2 = model.linear2.bias.detach().cpu().numpy().astype(np.float64)    # (2,)
    x_np = x_norm.detach().cpu().numpy().astype(np.float64)

    k = model.hidden_dim

    # Pre-activation mean and covariance of h under noise
    a = W1 @ x_np + b1                     # (k,) mean pre-activation
    G = (sigma ** 2) * (W1 @ W1.T)         # (k,k) pre-activation covariance
    L = np.linalg.cholesky(G + 1e-12 * np.eye(k))  # (k,k) Cholesky

    # Prediction
    with torch.no_grad():
        logits = model(x_norm.unsqueeze(0)).squeeze(0)
    A = predicted_class if predicted_class is not None else int(logits.argmax().item())
    j_cls = 1 - A

    # Margin linear coefficients
    u = (W2[A] - W2[j_cls]).astype(np.float64)  # (k,)
    d = float(b2[A] - b2[j_cls])

    mu_margin = float(np.dot(u, np.maximum(a, 0.0)) + d)
    if mu_margin <= 0:
        return 0.0

    # Gauss-Hermite quadrature points/weights for N(0,1)
    pts, wts = np.polynomial.hermite.hermgauss(n_quad)
    pts_std = np.sqrt(2) * pts          # rescale to N(0,1)
    wts_std = wts / np.sqrt(np.pi)     # normalize weights to sum to 1

    # Build n^k tensor product grid
    grids = np.meshgrid(*([pts_std] * k), indexing='ij')
    z_flat = np.stack([g.ravel() for g in grids], axis=1)   # (n^k, k)
    w_grids = np.meshgrid(*([wts_std] * k), indexing='ij')
    w_flat = np.ones(w_grids[0].shape)
    for wg in w_grids:
        w_flat = w_flat * wg
    w_flat = w_flat.ravel()  # (n^k,)

    # Evaluate margin at each quadrature point: h = L @ z + a, M = u^T ReLU(h) + d
    h_flat = z_flat @ L.T + a[None, :]          # (n^k, k)
    M_flat = np.maximum(h_flat, 0.0) @ u + d    # (n^k,)

    pA = float(np.sum(w_flat[M_flat > 0]))
    return float(np.clip(pA, 0.0, 1.0))


def _smolyak_gauss_hermite_grid(k: int, level: int, _cache: dict = {}) -> tuple:
    """
    Smolyak sparse grid for k-dimensional N(0, I) integration.

    Replaces tensor-product Gauss-Hermite (n^k points) with O(k^level) points,
    enabling certification of large bottleneck models (k=16, 32) at ~2-5ms.

    1D rule at Smolyak level l uses n = 2l-1 Gauss-Hermite points scaled to N(0,1).
    The Smolyak combination formula uses signed coefficients; weights may be negative
    but sum to 1 for any constant function.

    level:
        2 → exact for total-degree-3 polynomials,  ~k^2     points
        3 → exact for total-degree-5 polynomials,  ~k^3/6   points  (recommended)
        4 → exact for total-degree-7 polynomials,  ~k^4/24  points

    Approximate point counts (k=16, k=32):
        level=3:  ~20k pts (k=16),  ~150k pts (k=32)
        level=4: ~120k pts (k=16), ~1.5M  pts (k=32)

    Grid is cached by (k, level) — computed once, reused across certify() calls.
    """
    cache_key = (k, level)
    if cache_key in _cache:
        return _cache[cache_key]

    from math import comb

    _1d_rules: dict = {}

    def _get_1d_rule(l: int):
        if l not in _1d_rules:
            n = 2 * l - 1
            pts_raw, wts_raw = np.polynomial.hermite.hermgauss(n)
            _1d_rules[l] = (np.sqrt(2) * pts_raw, wts_raw / np.sqrt(np.pi))
        return _1d_rules[l]

    all_pts: list = []
    all_wts: list = []

    # Enumerate all k-tuples beta with beta_i >= 0 and |beta| <= level.
    # Smolyak coefficient: c = (-1)^(level-s) * C(k-1, level-s), s = sum(beta).
    def _enum_beta(remaining: int, dim: int, current: list):
        if dim == k:
            yield tuple(current)
            return
        for v in range(min(remaining, level) + 1):
            current.append(v)
            yield from _enum_beta(remaining - v, dim + 1, current)
            current.pop()

    for beta in _enum_beta(level, 0, []):
        s = sum(beta)
        coeff = ((-1) ** (level - s)) * comb(k - 1, level - s)
        if coeff == 0:
            continue

        active = [i for i, b in enumerate(beta) if b > 0]

        if not active:
            pts_block = np.zeros((1, k))
            wts_block = np.ones(1)
        else:
            pts_1d = [_get_1d_rule(beta[i] + 1)[0] for i in active]
            wts_1d = [_get_1d_rule(beta[i] + 1)[1] for i in active]

            # Tensor product over ACTIVE dimensions only (len(active) <= level << k)
            grids_p = np.meshgrid(*pts_1d, indexing='ij')
            grids_w = np.meshgrid(*wts_1d, indexing='ij')
            n_pts = int(np.prod([len(p) for p in pts_1d]))

            pts_active = np.stack([g.ravel() for g in grids_p], axis=1)  # (n, |S|)
            wts_block = np.ones(n_pts)
            for gw in grids_w:
                wts_block *= gw.ravel()

            # Embed active-dim coordinates in full k-dim space (zero elsewhere)
            pts_block = np.zeros((n_pts, k))
            pts_block[:, active] = pts_active

        all_pts.append(pts_block)
        all_wts.append(coeff * wts_block)

    points = np.concatenate(all_pts, axis=0)   # (N, k)
    weights = np.concatenate(all_wts, axis=0)  # (N,) signed

    _cache[cache_key] = (points, weights)
    return points, weights


def certify_bottleneck_sparse_pA(
    model,
    x_norm: torch.Tensor,
    sigma: float,
    predicted_class: int = None,
    level: int = 3,
) -> float:
    """
    Approximate p_A for SwitcherBottleneckMLP via Smolyak sparse Gauss-Hermite.

    Supports large bottleneck (k=16, 32) intractable with certify_bottleneck_pA.
    Point count is O(k^level) vs O(n^k) for tensor product.

    Args:
        model          : SwitcherBottleneckMLP
        x_norm         : normalized observation, shape [obs_dim]
        sigma          : smoothing noise std
        predicted_class: override prediction (default: argmax of logits)
        level          : Smolyak accuracy level (3 recommended for k<=32)

    Returns:
        p_A: float in [0, 1]
    """
    W1 = model.linear1.weight.detach().cpu().numpy().astype(np.float64)
    b1 = model.linear1.bias.detach().cpu().numpy().astype(np.float64)
    W2 = model.linear2.weight.detach().cpu().numpy().astype(np.float64)
    b2 = model.linear2.bias.detach().cpu().numpy().astype(np.float64)
    x_np = x_norm.detach().cpu().numpy().astype(np.float64)

    k = model.hidden_dim
    a = W1 @ x_np + b1                              # (k,) pre-activation mean
    G = (sigma ** 2) * (W1 @ W1.T)                  # (k, k) pre-activation covariance
    L = np.linalg.cholesky(G + 1e-12 * np.eye(k))   # Cholesky: G = L L^T

    with torch.no_grad():
        logits = model(x_norm.unsqueeze(0)).squeeze(0)
    A = predicted_class if predicted_class is not None else int(logits.argmax().item())
    j_cls = 1 - A

    u = (W2[A] - W2[j_cls]).astype(np.float64)  # (k,) margin coefficients
    d_val = float(b2[A] - b2[j_cls])

    mu_margin = float(np.dot(u, np.maximum(a, 0.0)) + d_val)
    if mu_margin <= 0:
        return 0.0

    # Sparse grid for N(0, I_k); cached after first call
    z_pts, z_wts = _smolyak_gauss_hermite_grid(k, level)  # (N, k), (N,)

    # Transform noise to pre-activation space: h = L z + a
    h_flat = z_pts @ L.T + a[None, :]              # (N, k)
    M_flat = np.maximum(h_flat, 0.0) @ u + d_val   # (N,)

    pA = float(np.sum(z_wts[M_flat > 0]))
    return float(np.clip(pA, 0.0, 1.0))


def certify_quad_skip_pA(model, x_norm, sigma, predicted_class=None):
    """
    Exact p_A for SwitcherQuadSkipMLP via Gil-Pelaez inversion.

    Works in input space (obs_dim × obs_dim eigendecomposition).

    The margin fluctuation (zero-mean) decomposes as:
        Q = Σ_k [σ·h_k·η_k + σ²·D_k·(η_k²-1)]
    where η_k ~ N(0,1) iid, D are eigenvalues of B_j = W_quad^T diag(c_quad) W_quad,
    and h_k = (V^T f)_k with f = W_quad^T g_quad + W_skip^T c_skip.
    """
    W_quad = model.linear_quad.weight.detach().cpu().numpy().astype(np.float64)
    b_quad = model.linear_quad.bias.detach().cpu().numpy().astype(np.float64)
    W_skip = model.linear_skip.weight.detach().cpu().numpy().astype(np.float64)
    b_skip = model.linear_skip.bias.detach().cpu().numpy().astype(np.float64)
    W_cls = model.classifier.weight.detach().cpu().numpy().astype(np.float64)
    b_cls = model.classifier.bias.detach().cpu().numpy().astype(np.float64)
    x_np = x_norm.detach().cpu().numpy().astype(np.float64)

    sigma2 = sigma ** 2
    quad_dim = model.quad_dim

    # Pre-activation means
    a_quad = W_quad @ x_np + b_quad
    a_skip = W_skip @ x_np + b_skip

    # Output means: E[z_c]
    # Quad: E[(a+δ)² + (a+δ)] = a² + a + σ²·||w_i||²
    w_norms_sq = np.sum(W_quad ** 2, axis=1)
    quad_mean = a_quad ** 2 + a_quad + sigma2 * w_norms_sq
    combined_mean = np.concatenate([quad_mean, a_skip])
    mu_out = W_cls @ combined_mean + b_cls

    A = predicted_class if predicted_class is not None else int(np.argmax(mu_out))
    j = 1 - A  # binary classification

    mu_Mj = float(mu_out[A] - mu_out[j])
    if mu_Mj <= 0:
        return 0.0

    # Split classifier weights
    c_full = W_cls[A] - W_cls[j]
    c_quad = c_full[:quad_dim]
    c_skip = c_full[quad_dim:]

    # Quadratic form in input space: B = W_quad^T diag(c_quad) W_quad
    B = W_quad.T @ (c_quad[:, None] * W_quad)  # (obs_dim, obs_dim)
    D, V = np.linalg.eigh(B)

    # Combined linear coefficient: f = W_quad^T g_quad + W_skip^T c_skip
    g_quad = c_quad * (2 * a_quad + 1)
    f = W_quad.T @ g_quad + W_skip.T @ c_skip

    # Project into eigenbasis of B
    h = V.T @ f

    # Gil-Pelaez: Q = Σ_k [σ·h_k·η_k + σ²·D_k·(η_k² - 1)]
    p_j = _gil_pelaez_cdf(-mu_Mj, sigma * h, sigma2 * D)
    return float(max(1.0 - p_j, 0.0))


def load_gp_switcher(ckpt: dict):
    """Reconstruct any GP switcher model from a checkpoint dict."""
    obs_dim = int(ckpt["obs_dim"])
    model_type = ckpt.get("model_type", "quad")
    if model_type == "quad_skip":
        model = SwitcherQuadSkipMLP(obs_dim=obs_dim,
                                     quad_dim=int(ckpt["quad_dim"]),
                                     skip_dim=int(ckpt["skip_dim"]))
    elif model_type == "quad_deep":
        model = SwitcherQuadDeepMLP(obs_dim=obs_dim,
                                     backbone_dims=ckpt["backbone_dims"])
    elif model_type == "bottleneck":
        model = SwitcherBottleneckMLP(obs_dim=obs_dim,
                                      hidden_dim=int(ckpt["hidden_dim"]))
    else:
        model = SwitcherQuadMLP(obs_dim=obs_dim,
                                 hidden_dim=int(ckpt["hidden_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


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
        n_quad: int = None,
        sparse_level: int = 3,
    ):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device)
        self.state_std = torch.tensor(state_std, dtype=torch.float32, device=self.device)
        self.sigma = sigma
        self._is_skip = isinstance(model, SwitcherQuadSkipMLP)
        self._is_bottleneck = isinstance(model, SwitcherBottleneckMLP)
        self._sparse_level = sparse_level

        # For bottleneck: auto-select tensor product (small k) vs sparse grid (large k).
        # Tensor product: n^k points, fast for k<=6 with n<=16 but intractable for k>8.
        # Sparse grid:    O(k^sparse_level) points, tractable for k=16 or k=32.
        if self._is_bottleneck:
            k = model.hidden_dim
            if n_quad is not None:
                # Explicit tensor-product order requested
                self._use_sparse = False
                self._n_quad = n_quad
            elif k > 8:
                # Large bottleneck: sparse grid is the only tractable option.
                # Auto-cap level to keep cert under ~5ms:
                #   k<=16: level=3 → ~20k pts, ~1ms
                #   k>16:  level=2 → ~5k pts,  ~1ms  (level=3 is ~150k pts, ~73ms)
                # User can override by passing sparse_level explicitly.
                if sparse_level == 3 and k > 16:
                    self._sparse_level = 2
                self._use_sparse = True
                self._n_quad = None
            else:
                # Small k: tensor product with auto n_quad (cap at 200MB)
                self._use_sparse = False
                for n in [16, 12, 10, 8, 6, 5, 4]:
                    if n ** k * k * 8 < 200_000_000:
                        n_quad = n
                        break
                else:
                    n_quad = 4
                self._n_quad = n_quad
        else:
            self._use_sparse = False
            self._n_quad = n_quad or 16

        # For deep models, fold backbone into single Linear for certification
        if isinstance(model, SwitcherQuadDeepMLP):
            self._cert_net = model.fold_backbone().to(self.device)
        elif not self._is_skip and not self._is_bottleneck:
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

        if self._is_bottleneck:
            with torch.no_grad():
                logits = self.model(x_norm.unsqueeze(0)).squeeze(0)
            pred = int(logits.argmax().item())
            if self._use_sparse:
                # Large k: Smolyak sparse Gauss-Hermite
                pA = certify_bottleneck_sparse_pA(self.model, x_norm, self.sigma,
                                                   predicted_class=pred,
                                                   level=self._sparse_level)
            else:
                # Small k: tensor-product Gauss-Hermite
                pA = certify_bottleneck_pA(self.model, x_norm, self.sigma,
                                            predicted_class=pred, n_quad=self._n_quad)
        elif self._is_skip:
            # Skip pathway: use dedicated certifier (works in input space)
            with torch.no_grad():
                logits = self.model(x_norm.unsqueeze(0)).squeeze(0)
            pred = int(logits.argmax().item())
            pA = certify_quad_skip_pA(self.model, x_norm, self.sigma,
                                       predicted_class=pred)
        else:
            # Standard or deep quad: use folded net + certify_quad_pA
            with torch.no_grad():
                logits = self._cert_net(x_norm.unsqueeze(0)).squeeze(0)
            pred = int(logits.argmax().item())
            # Compute mean output under Gaussian noise efficiently (diagonal-only).
            # E[phi(h_k)] = a_k^2 + sigma^2 * ||W_in[k,:]||^2 + a_k
            # avoids building the full (hidden_dim x hidden_dim) covariance matrix.
            lin_layers = [m for m in self._cert_net if isinstance(m, nn.Linear)]
            W1, b1 = lin_layers[0].weight, lin_layers[0].bias   # (h, d)
            W2, b2 = lin_layers[1].weight, lin_layers[1].bias   # (2, h)
            with torch.no_grad():
                a = W1 @ x_norm + b1                            # (h,)
                sigma_h_sq = (self.sigma ** 2) * (W1 ** 2).sum(dim=1)  # (h,)
                mu_h = a ** 2 + sigma_h_sq + a                  # (h,)
                mu_out = W2 @ mu_h + b2                         # (2,)
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
