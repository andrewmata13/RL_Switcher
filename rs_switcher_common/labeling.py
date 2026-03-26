"""
Adversarial-detection labeling dataset builder for MuJoCo environments.

Collects clean and adversarially-perturbed observations from PPO rollouts.
Clean obs -> y=0 (non-critical, use PPO).
Adversarial obs (opt_attack applied) -> y=1 (critical, use ATLA).

50/50 split by construction: every clean obs is paired with its attacked version.
"""
from typing import List, Tuple
import numpy as np

from .attacks import opt_attack


class CriticalBurstLabeler:
    """
    Builds a binary detection dataset from clean PPO rollouts:
      - clean obs              -> y=0 (non-critical, use PPO)
      - opt_attack(clean obs)  -> y=1 (critical, use ATLA)

    cfg is accepted for API compatibility but unused (eps comes from the policy).
    """

    def __init__(self, perf_policy, cfg=None, state_mean=None, state_std=None):
        self.perf = perf_policy
        self.eps = perf_policy.eps

    def build_dataset(
        self,
        n_episodes: int = 10,
        subsample_every: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect clean and adversarial observations from PPO rollouts.

        Returns
        -------
        X : (2*N, obs_dim)  interleaved clean / adversarial obs
        y : (2*N,)          0=clean, 1=adversarial
        """
        X: List[np.ndarray] = []
        Y: List[int] = []

        for ep in range(n_episodes):
            obs = self.perf.start_episode()
            done = False
            t = 0

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


def collect_state_stats(perf, episodes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Collect running mean/std of normalized observations over clean rollouts."""
    states = []
    for ep in range(episodes):
        obs = perf.start_episode()
        done = False
        while not done:
            states.append(obs.astype("float32"))
            obs, _, done, _ = perf.step(perf.predict(obs))
    states = np.stack(states)
    return states.mean(axis=0), states.std(axis=0) + 1e-6
