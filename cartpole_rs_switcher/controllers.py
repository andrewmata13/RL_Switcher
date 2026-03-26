import numpy as np
import torch
import gymnasium as gym
from scipy.linalg import solve_discrete_are
from stable_baselines3 import PPO


class QuantizedLQRBackup:
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.K = self._design_lqr_gain()

    def _design_lqr_gain(self) -> np.ndarray:
        g = 9.8
        mc = 1.0
        mp = 0.1
        total_mass = mc + mp
        l = 0.5

        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -(mp * g) / mc, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, (total_mass * g) / (l * mc), 0.0],
            ],
            dtype=np.float64,
        )
        B = np.array([[0.0], [1.0 / mc], [0.0], [-1.0 / (l * mc)]], dtype=np.float64)

        Ad = np.eye(4) + self.dt * A
        Bd = self.dt * B
        Q = np.diag([1.0, 0.2, 20.0, 1.0])
        R = np.array([[0.1]], dtype=np.float64)
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
        return K.astype(np.float32)

    def predict(self, obs: np.ndarray) -> int:
        u_cont = float(-(self.K @ obs.reshape(-1, 1)).squeeze())
        return 1 if u_cont >= 0.0 else 0


class PerfPolicy:
    def __init__(self, model: PPO, device: torch.device):
        self.model = model
        self.device = device
        self.policy = model.policy
        self.policy.set_training_mode(False)

    @classmethod
    def load(cls, path: str, device: torch.device):
        model = PPO.load(path, device=device)
        return cls(model, device)

    def predict(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

