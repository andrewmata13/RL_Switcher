from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from .controllers import PerfPolicy, QuantizedLQRBackup
from .rs import VanillaRSSwitcher


class AlwaysPerfController:
    def __init__(self, perf: PerfPolicy):
        self.perf = perf

    def select(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        return self.perf.predict(obs), {
            "allow_perf": 1.0, "p_critical": 0.0, "p_allow": 1.0,
            "R_rs": np.nan, "R_exec": np.nan,
        }


class AlwaysBackupController:
    def __init__(self, backup: QuantizedLQRBackup):
        self.backup = backup

    def select(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        return self.backup.predict(obs), {
            "allow_perf": 0.0, "p_critical": 1.0, "p_allow": 0.0,
            "R_rs": np.nan, "R_exec": np.nan,
        }


class UncertifiedSwitcherController:
    """Single forward pass through the switcher — no smoothing, no radius gate."""

    def __init__(self, perf: PerfPolicy, backup: QuantizedLQRBackup, rs: VanillaRSSwitcher):
        self.perf = perf
        self.backup = backup
        self.rs = rs

    def select(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        pred = self.rs.predict(obs)  # 0=non-critical, 1=critical
        allow_perf = pred == 0
        action = self.perf.predict(obs) if allow_perf else self.backup.predict(obs)
        return action, {
            "allow_perf": float(allow_perf),
            "p_critical": float(pred), "p_allow": float(1 - pred),
            "R_rs": np.nan, "R_exec": np.nan,
        }


class CertifiedSwitcherController:
    """MC randomized smoothing: allows perf only when certified non-critical with R >= delta_budget_l2."""

    def __init__(self, perf: PerfPolicy, backup: QuantizedLQRBackup, rs: VanillaRSSwitcher,
                 delta_budget_l2: float):
        self.perf = perf
        self.backup = backup
        self.rs = rs
        self.delta_budget_l2 = delta_budget_l2

    def select(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        pred, p_A_lower, R = self.rs.certify(obs)
        allow_perf = (pred == 0) and (R >= self.delta_budget_l2)
        action = self.perf.predict(obs) if allow_perf else self.backup.predict(obs)
        p_critical = 1.0 - p_A_lower if pred == 0 else p_A_lower
        return action, {
            "allow_perf": float(allow_perf),
            "p_critical": p_critical, "p_allow": 1.0 - p_critical,
            "p_A_lower": p_A_lower,
            "R_rs": R, "R_exec": R,
        }


def evaluate_controller(env_id: str, controller, episodes: int = 20, seed: int = 0):
    env = gym.make(env_id)
    returns: List[float] = []
    logs: List[Dict[str, float]] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action, info = controller.select(np.array(obs, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            logs.append(info)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return returns, logs
