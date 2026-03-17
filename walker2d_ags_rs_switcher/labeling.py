"""
Adversarial-detection labeling dataset for Walker2D.

Label design (same as Hopper/HalfCheetah):
  y=0  clean observation from PPO rollout
  y=1  opt_attack(obs) — adversarially perturbed observation

50/50 split by construction.
"""
from typing import List, Tuple
import numpy as np

from .attacks import opt_attack
from .controllers import Walker2DPerfPolicy


class Walker2DCriticalBurstLabeler:
    def __init__(self, perf_policy: Walker2DPerfPolicy, cfg, state_mean, state_std):
        self.perf = perf_policy
        self.eps  = perf_policy.eps

    def build_dataset(
        self,
        n_episodes: int = 10,
        subsample_every: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X: List[np.ndarray] = []
        Y: List[int]        = []

        for ep in range(n_episodes):
            obs  = self.perf.start_episode()
            done = False
            t    = 0

            while not done:
                if t % subsample_every == 0:
                    X.append(obs.copy())
                    Y.append(0)
                    adv = opt_attack(self.perf.attack_model, obs, eps=self.eps)
                    X.append(adv.copy())
                    Y.append(1)

                action = self.perf.predict(obs)
                obs, _, done, _ = self.perf.step(action)
                t += 1

        return np.stack(X).astype(np.float32), np.array(Y, dtype=np.int64)
