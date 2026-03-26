"""
Unified MuJoCo policy wrappers parameterized by EnvConfig.

MuJoCoPerfPolicy  — native PPO (high performance, vulnerable to attack)
MuJoCoBackupPolicy — ATLA defended policy (robust, used as safe fallback)

CtsPolicy is imported from other_attacks/optimal_attack/opt_pg/models.py.
"""
import numpy as np
import torch

from .compat import ensure_paths, patch_gym_env
from .env_config import EnvConfig

# Ensure paths are set up before importing CtsPolicy
ensure_paths()
from other_attacks.optimal_attack.opt_pg.models import CtsPolicy  # noqa: E402


def raw_obs_from_sim(custom_env, config: EnvConfig) -> np.ndarray:
    """
    Extract raw observation from the MuJoCo simulator state.

    Reads qpos and qvel slices according to config, optionally clips qvel.
    """
    inner = custom_env.env.unwrapped
    qpos = np.array(inner.sim.data.qpos.flat[config.qpos_slice[0]:config.qpos_slice[1]])
    qvel = np.array(inner.sim.data.qvel.flat[config.qvel_slice[0]:config.qvel_slice[1]])
    if config.qvel_clip is not None:
        qvel = np.clip(qvel, -config.qvel_clip, config.qvel_clip)
    return np.concatenate([qpos, qvel]).astype(np.float32)


class MuJoCoPerfPolicy:
    """
    Native PPO policy for MuJoCo environments.  Loaded from the attack
    checkpoint so that the policy, ZFilter, and adversary all share
    consistent statistics.
    """

    def __init__(self, model: CtsPolicy, custom_env, config: EnvConfig,
                 attack_model=None):
        self.model = model
        self.custom_env = custom_env
        self.config = config
        self.attack_model = attack_model
        self.eps = config.eps

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normalize raw obs using the frozen filter (from the last reset)."""
        from copy import deepcopy
        f = getattr(self.custom_env, "new_filter", self.custom_env.state_filter)
        return deepcopy(f)(raw_obs.astype(np.float32)).astype(np.float32)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        """obs_norm: normalized obs. Returns action clipped to [-1,1]."""
        with torch.no_grad():
            t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            pds = self.model(t)
            act = torch.clamp(pds[0], -1.0, 1.0)
        return act.numpy()[0].astype(np.float32)

    def start_episode(self, max_steps: int = 1000) -> np.ndarray:
        """
        Reset to a random gym starting state using the custom_env API.
        Returns the initial normalized observation.
        """
        self._step_count = 0
        self._max_steps = max_steps

        raw = self.custom_env.env.reset()
        if isinstance(raw, tuple):   # gym 0.26 returns (obs, info)
            raw = raw[0]

        inner = self.custom_env.env.unwrapped
        qpos = inner.sim.data.qpos.copy()
        qvel = inner.sim.data.qvel.copy()
        uState = np.concatenate([qpos, qvel]).astype(np.float32)

        obs_norm = self.custom_env.reset(uState, None, name=self.config.name)
        return obs_norm.astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Step via custom_env.step() (handles filter + custom termination).
        Returns (obs_norm, norm_reward, is_done, info).
        Episode ends after max_steps even if the agent stays upright.
        """
        result, norm_rew, is_done, info = self.custom_env.step(
            action, change_filter=False, name=self.config.name
        )
        obs_norm = result[1].astype(np.float32)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            is_done = True
        return obs_norm, norm_rew, is_done, info

    @classmethod
    def load(cls, config: EnvConfig, model_path: str,
             attack_path: str = None, naive_policy: bool = False):
        """
        Load PPO and (optionally) the adversary.

        Parameters
        ----------
        config : EnvConfig
            Environment configuration (obs_dim, action_dim, name, eps).
        model_path : str
            Path to the PPO checkpoint.
        attack_path : str, optional
            Path to the co-trained attack checkpoint.
        naive_policy : bool
            If True, load policy_model from model_path while taking custom_env
            and adversary_model from attack_path.
        """
        src = attack_path if attack_path is not None else model_path
        ck = torch.load(src, map_location="cpu")

        if naive_policy and attack_path is not None:
            ck_naive = torch.load(model_path, map_location="cpu")
            policy_ck = ck_naive
            env_ck = ck_naive
        else:
            policy_ck = ck
            env_ck = ck

        model = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
        model.load_state_dict(policy_ck["policy_model"])
        model.log_stdev.data[:] = -100
        model.eval()

        custom_env = env_ck["envs"][0]
        custom_env.normalizer_read_only = True
        patch_gym_env(custom_env.env)

        attack_model = None
        if attack_path is not None:
            attack_model = CtsPolicy(config.obs_dim, config.obs_dim, "orthogonal")
            attack_model.load_state_dict(ck["adversary_policy_model"])
            attack_model.log_stdev.data[:] = -100
            attack_model.eval()

        return cls(model, custom_env, config, attack_model=attack_model)


class MuJoCoBackupPolicy:
    """
    ATLA-defended backup policy.  Uses its own ZFilter for normalisation.
    """

    def __init__(self, model: CtsPolicy, custom_env, config: EnvConfig):
        self.model = model
        self.custom_env = custom_env
        self.config = config

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        """Apply ATLA's read-only ZFilter to a raw observation."""
        return self.custom_env.state_filter(
            raw_obs.astype(np.float32)
        ).astype(np.float32)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            pds = self.model(t)
            act = torch.clamp(pds[0], -1.0, 1.0)
        return act.numpy()[0].astype(np.float32)

    @classmethod
    def load(cls, config: EnvConfig, model_path: str):
        ck = torch.load(model_path, map_location="cpu")

        model = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
        model.load_state_dict(ck["policy_model"])
        model.log_stdev.data[:] = -100
        model.eval()

        custom_env = ck["envs"][0]
        custom_env.normalizer_read_only = True
        patch_gym_env(custom_env.env)

        return cls(model, custom_env, config)
