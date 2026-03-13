from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from .attacks import pgd_l2_attack
from .config import LabelConfig
from .controllers import PerfPolicy


class CriticalBurstLabeler:
    def __init__(self, env_id: str, perf_policy: PerfPolicy, cfg: LabelConfig,
                 state_mean: np.ndarray, state_std: np.ndarray):
        self.env_id = env_id
        self.perf_policy = perf_policy
        self.cfg = cfg
        self.state_mean = state_mean
        self.state_std = state_std

    def clone_env_at_state(self, state: np.ndarray) -> gym.Env:
        env = gym.make(self.env_id)
        env.reset(seed=self.cfg.seed)
        env.unwrapped.state = np.array(state, dtype=np.float32).copy()
        return env

    def rollout_from_state(self, state: np.ndarray, attack_start: Optional[int]) -> Tuple[float, bool]:
        env = self.clone_env_at_state(state)
        obs = np.array(state, dtype=np.float32).copy()
        total_reward = 0.0
        failed = False

        for t in range(self.cfg.horizon_h):
            if attack_start is not None and attack_start <= t < attack_start + self.cfg.burst_k:
                obs_used = pgd_l2_attack(self.perf_policy, obs, self.cfg.epsilon_l2,
                                        self.state_mean, self.state_std, n_steps=self.cfg.pgd_steps)
            else:
                obs_used = obs
            act = self.perf_policy.predict(obs_used)
            obs, reward, terminated, truncated, _ = env.step(act)
            total_reward += reward
            if terminated or truncated:
                failed = True
                break
        env.close()
        return total_reward, failed

    def label_state(self, state: np.ndarray) -> int:
        clean_return, _ = self.rollout_from_state(state, attack_start=None)
        max_burst_start = max(0, self.cfg.horizon_h - self.cfg.burst_k)
        starts = np.linspace(0, max_burst_start, self.cfg.n_attack_starts, dtype=int)
        for start in starts:
            attacked_return, attacked_fail = self.rollout_from_state(state, attack_start=int(start))
            if attacked_fail or (clean_return - attacked_return) >= self.cfg.reward_drop_threshold:
                return 1
        return 0

    def build_dataset(self, perf_model: PPO, n_episodes: int = 30, subsample_every: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        env = gym.make(self.env_id)
        X: List[np.ndarray] = []
        Y: List[int] = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=self.cfg.seed + ep)
            done = False
            t = 0
            while not done:
                act, _ = perf_model.predict(obs, deterministic=True)
                next_obs, _, terminated, truncated, _ = env.step(act)
                if t % subsample_every == 0:
                    X.append(np.array(obs, dtype=np.float32))
                    Y.append(self.label_state(np.array(obs, dtype=np.float32)))
                obs = next_obs
                t += 1
                done = terminated or truncated
        env.close()
        return np.stack(X), np.array(Y, dtype=np.int64)
