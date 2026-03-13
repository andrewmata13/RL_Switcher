import math
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from .models import SwitcherMLP
from .utils import normalize


@dataclass
class AGSRegionParams:
    alpha: np.ndarray
    beta: float
    tau2: float
    lower: np.ndarray
    upper: np.ndarray


class RegionizedAGSSwitcher:
    def __init__(self, model: SwitcherMLP, state_mean: np.ndarray, state_std: np.ndarray, bins_per_dim: int = 3):
        self.model = model.eval()
        self.obs_dim = model.fc1.in_features
        self.hidden_dim = model.fc1.out_features
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.bins_per_dim = bins_per_dim

        self.W1 = model.fc1.weight.detach().cpu().numpy().copy()
        self.b1 = model.fc1.bias.detach().cpu().numpy().copy()
        self.w2 = model.fc2.weight.detach().cpu().numpy().reshape(-1).copy()
        self.d2 = float(model.fc2.bias.detach().cpu().numpy()[0])

        self.regions: Dict[Tuple[int, ...], AGSRegionParams] = {}
        self.edges: List[np.ndarray] = []

    def _build_edges(self, X_norm: np.ndarray) -> None:
        self.edges = []
        qs = np.linspace(0.0, 1.0, self.bins_per_dim + 1)
        for d in range(self.obs_dim):
            e = np.quantile(X_norm[:, d], qs)
            for i in range(1, len(e)):
                if e[i] <= e[i - 1]:
                    e[i] = e[i - 1] + 1e-6
            self.edges.append(e)

    def _region_index(self, x_norm: np.ndarray) -> Tuple[int, ...]:
        idx = []
        for d in range(self.obs_dim):
            e = self.edges[d]
            i = int(np.clip(np.digitize(x_norm[d], e[1:-1], right=False), 0, self.bins_per_dim - 1))
            idx.append(i)
        return tuple(idx)

    def _region_bounds(self, region_idx: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.obs_dim, dtype=np.float32)
        upper = np.zeros(self.obs_dim, dtype=np.float32)
        for d, i in enumerate(region_idx):
            lower[d] = self.edges[d][i]
            upper[d] = self.edges[d][i + 1]
        return lower, upper

    def fit(self, X: np.ndarray) -> None:
        X_norm = normalize(X, self.state_mean, self.state_std)
        self._build_edges(X_norm)

        region_points: Dict[Tuple[int, ...], List[np.ndarray]] = {}
        for x in X_norm:
            idx = self._region_index(x)
            region_points.setdefault(idx, []).append(x)

        Z_all = X_norm @ self.W1.T + self.b1[None, :]
        mu_global = Z_all.mean(axis=0)
        var_global = Z_all.var(axis=0) + 1e-8

        for idx in product(range(self.bins_per_dim), repeat=self.obs_dim):
            pts = np.array(region_points.get(idx, []), dtype=np.float32)
            if len(pts) < max(10, self.hidden_dim // 2):
                mu_z = mu_global
                var_z = var_global
            else:
                Z = pts @ self.W1.T + self.b1[None, :]
                mu_z = Z.mean(axis=0)
                var_z = Z.var(axis=0) + 1e-8

            std_z = np.sqrt(var_z)
            alpha = mu_z / std_z
            A_diag = norm.cdf(alpha)
            c = std_z * norm.pdf(alpha)
            second_moment = (mu_z * mu_z + var_z) * norm.cdf(alpha) + mu_z * std_z * norm.pdf(alpha)
            mean_relu = A_diag * mu_z + c
            var_relu = np.maximum(second_moment - mean_relu * mean_relu, 1e-10)
            var_eps = np.maximum(var_relu - (A_diag * A_diag) * var_z, 1e-10)

            alpha_r = self.W1.T @ (A_diag * self.w2)
            beta_r = float(np.dot(self.w2, A_diag * self.b1 + c) + self.d2)
            tau2_r = float(np.sum((self.w2 ** 2) * var_eps))
            lower, upper = self._region_bounds(idx)
            self.regions[idx] = AGSRegionParams(alpha=alpha_r, beta=beta_r, tau2=tau2_r, lower=lower, upper=upper)

    def region_params(self, obs: np.ndarray) -> AGSRegionParams:
        x_norm = normalize(obs, self.state_mean, self.state_std)
        idx = self._region_index(x_norm)
        return self.regions[idx]

    def region_radius(self, obs: np.ndarray) -> float:
        x_norm = normalize(obs, self.state_mean, self.state_std)
        params = self.region_params(obs)
        d_norm = np.minimum(x_norm - params.lower, params.upper - x_norm)
        d_norm = np.maximum(d_norm, 0.0)
        return float(np.min(d_norm * self.state_std))

    def smoothed_prob_and_radius(self, obs: np.ndarray, sigma: float):
        params = self.region_params(obs)
        x_norm = normalize(obs, self.state_mean, self.state_std)
        m = float(np.dot(params.alpha, x_norm) + params.beta)
        v = float((sigma ** 2) * np.dot(params.alpha, params.alpha) + params.tau2)
        std = math.sqrt(max(v, 1e-12))
        p1 = float(norm.cdf(m / std))
        p0 = 1.0 - p1
        pA = max(p0, p1)
        pB = 1.0 - pA
        R_rs = 0.5 * sigma * (norm.ppf(pA) - norm.ppf(pB))
        return p1, p0, pA, pB, float(R_rs)
