"""
Clean policy wrappers using frozen (fixed) observation normalization.

Replaces MuJoCoPerfPolicy / MuJoCoBackupPolicy for environments trained with
the clean pipeline (train_clean_ppo.py + train_atla_l2.py).

Key difference: both PPO and ATLA share the same frozen mean/std computed once
from clean rollouts. No ZFilter ever updates during inference, eliminating all
ZFilter churn and the instability it causes in terminating environments.

Interface exactly matches MuJoCoPerfPolicy / MuJoCoBackupPolicy so that
evaluate_controller() works without modification.
"""
import numpy as np
import torch
import gymnasium as gym

from .env_config import EnvConfig
from .compat import ensure_paths

ensure_paths()
from other_attacks.optimal_attack.opt_pg.models import CtsPolicy  # noqa: E402

# Map EnvConfig.name -> gymnasium env ID
_GYM_IDS = {
    "Hopper":   "Hopper-v4",
    "Cheetah":  "HalfCheetah-v4",
    "Walker2D": "Walker2d-v4",
}


class _CleanEnvAdapter:
    """
    Thin shim so that evaluate_controller() and raw_obs_from_sim() work
    with a raw gymnasium env without modification.

    Mirrors the custom_env attributes accessed by the evaluation loop:
      .env             — raw gymnasium env (.unwrapped.data.qpos / .data.qvel)
      .total_true_reward — cumulative undiscounted episode reward (updated by CleanPerfPolicy.step)
    """
    def __init__(self, gym_env: gym.Env):
        self.env = gym_env
        self.total_true_reward = 0.0


class CleanPerfPolicy:
    """
    PPO policy with frozen obs normalization. Drop-in for MuJoCoPerfPolicy.

    Checkpoint format (saved by train_clean_ppo.py):
        {policy_model, adversary_policy_model (optional), norm_mean, norm_std}
    """

    def __init__(self, model: CtsPolicy, norm_mean: np.ndarray,
                 norm_std: np.ndarray, gym_env: gym.Env,
                 config: EnvConfig, attack_model: CtsPolicy = None):
        self.model = model
        self.norm_mean = norm_mean.astype(np.float32)
        self.norm_std  = norm_std.astype(np.float32)
        self.config = config
        self.eps = config.eps
        self.attack_model = attack_model
        self._env = gym_env
        self._adapter = _CleanEnvAdapter(gym_env)
        self._step_count = 0
        self._max_steps = 1000

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------
    @property
    def custom_env(self) -> _CleanEnvAdapter:
        return self._adapter

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        return ((raw_obs - self.norm_mean) / (self.norm_std + 1e-8)).astype(np.float32)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            act = torch.clamp(self.model(t)[0], -1.0, 1.0)
        return act.numpy()[0].astype(np.float32)

    def start_episode(self, max_steps: int = 1000) -> np.ndarray:
        self._max_steps = max_steps
        self._step_count = 0
        self._adapter.total_true_reward = 0.0
        obs, _ = self._env.reset()
        return self.normalize(obs)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        self._adapter.total_true_reward += float(reward)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            done = True
        return self.normalize(obs), float(reward), done, info

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, config: EnvConfig, ckpt_path: str,
             attack_path: str = None) -> "CleanPerfPolicy":
        """
        Load PPO policy from a clean checkpoint.

        attack_path: if provided, loads the adversary_policy_model from that
                     file (may be the same file as ckpt_path if trained together).
        """
        ck = torch.load(ckpt_path, map_location="cpu")

        model = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
        model.load_state_dict(ck["policy_model"])
        model.log_stdev.data[:] = -100
        model.eval()

        norm_mean = np.array(ck["norm_mean"], dtype=np.float32)
        norm_std  = np.array(ck["norm_std"],  dtype=np.float32)

        attack_model = None
        if attack_path is not None:
            ack = torch.load(attack_path, map_location="cpu")
            attack_model = CtsPolicy(config.obs_dim, config.obs_dim, "orthogonal")
            attack_model.load_state_dict(ack["adversary_policy_model"])
            attack_model.log_stdev.data[:] = -100
            attack_model.eval()

        gym_env = gym.make(_GYM_IDS[config.name])
        return cls(model, norm_mean, norm_std, gym_env, config, attack_model)


class PPOAsBackup:
    """
    Uses the PPO policy itself as the "backup" in the continuous controller.

    Key insight: in the clean pipeline, the backup always receives clean (unperturbed)
    simulator state via raw_obs_from_sim(). The adversary only perturbs obs_ppo
    (what the certifier and perf policy see), never obs_atla (the raw sim state).

    Therefore, running PPO on the unperturbed raw state:
      1. Produces the correct PPO action → attack is neutralized
      2. Stays on the PPO gait trajectory → ATLA→PPO transition fall rate = 0%

    This is the ideal backup when the only threat is observation perturbation
    (not environment state manipulation).

    Drop-in replacement for CleanBackupPolicy in ContinuousSwitcherController.
    """

    def __init__(self, perf: "CleanPerfPolicy"):
        self._perf = perf

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        return self._perf.normalize(raw_obs)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        return self._perf.predict(obs_norm)


class DegradedPPOBackup:
    """
    PPO with inference-time action noise. Provides a backup that is clearly weaker
    than PPO on clean observations while remaining gait-compatible (same gait period
    and phase → 0% transition fall rate).

    The backup receives clean simulator state (via raw_obs_from_sim), computes the
    correct PPO action, then adds Gaussian noise to the output. This degrades return
    smoothly with action_noise_sigma while preserving hopping stability.
    """

    def __init__(self, perf: "CleanPerfPolicy", action_noise_sigma: float = 0.2):
        self._perf = perf
        self._sigma = action_noise_sigma

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        return self._perf.normalize(raw_obs)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        action = self._perf.predict(obs_norm)
        noise = np.random.normal(0, self._sigma, action.shape).astype(np.float32)
        return np.clip(action + noise, -1.0, 1.0)


class CleanBackupPolicy:
    """
    ATLA backup policy using the same frozen normalization as CleanPerfPolicy.
    Drop-in for MuJoCoBackupPolicy.

    Checkpoint format (saved by train_atla_l2.py):
        {policy_model, norm_mean, norm_std}
    """

    def __init__(self, model: CtsPolicy, norm_mean: np.ndarray,
                 norm_std: np.ndarray, config: EnvConfig):
        self.model = model
        self.norm_mean = norm_mean.astype(np.float32)
        self.norm_std  = norm_std.astype(np.float32)
        self.config = config

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        return ((raw_obs - self.norm_mean) / (self.norm_std + 1e-8)).astype(np.float32)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            act = torch.clamp(self.model(t)[0], -1.0, 1.0)
        return act.numpy()[0].astype(np.float32)

    @classmethod
    def load(cls, config: EnvConfig, ckpt_path: str) -> "CleanBackupPolicy":
        """
        Load ATLA backup. The checkpoint must contain norm_mean / norm_std
        (copied from the clean PPO checkpoint during train_atla_l2.py).
        """
        ck = torch.load(ckpt_path, map_location="cpu")

        model = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
        model.load_state_dict(ck["policy_model"])
        model.log_stdev.data[:] = -100
        model.eval()

        norm_mean = np.array(ck["norm_mean"], dtype=np.float32)
        norm_std  = np.array(ck["norm_std"],  dtype=np.float32)

        return cls(model, norm_mean, norm_std, config)
