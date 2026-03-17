"""
Adversarial-detection labeling dataset for Hopper.

For Hopper, the pre-trained adversary (opt_attack) is powerful enough to cause
catastrophic failure from any state.  The discriminating signal is therefore the
OBSERVATION itself: a clean observation (from normal PPO rollout) is non-critical
(y=0), while an adversarially-perturbed observation is critical (y=1).

The switcher is trained to detect the presence of adversarial perturbation.
At runtime, RS certification checks that the current obs is robustly classified
as "non-adversarial" before allowing the high-performance PPO.
"""
from typing import List, Tuple
import numpy as np

from .attacks import opt_attack
from .controllers import HopperPerfPolicy


class HopperCriticalBurstLabeler:
    """
    Builds a binary detection dataset from clean PPO rollouts:
      - clean obs              → y=0 (non-critical, use PPO)
      - opt_attack(clean obs)  → y=1 (critical, use ATLA)

    The name keeps backwards-compatibility with build_labels_hopper.py.
    cfg is accepted for API compatibility but unused (eps comes from the policy).
    """

    def __init__(self, perf_policy: HopperPerfPolicy, cfg, state_mean, state_std):
        self.perf = perf_policy
        self.eps  = perf_policy.eps

    def build_dataset(
        self,
        n_episodes: int = 10,
        subsample_every: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect clean and adversarial observations from PPO rollouts.

        Returns
        -------
        X : (2*N, 11)  interleaved clean / adversarial obs
        y : (2*N,)     0=clean, 1=adversarial
        """
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
