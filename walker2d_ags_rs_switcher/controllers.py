"""
Walker2D policy wrappers.

Mirrors halfcheetah_ags_rs_switcher/controllers.py with Walker2D-specific params:
  - obs_dim=17, action_dim=6
  - name="Walker2D" in all custom_env calls
  - uState for reset: qpos(9) + qvel(9) = 18D; obs = uState[1:] = 17D
  - _raw_obs_from_sim: qpos[1:9] + clip(qvel[:9], -10, 10)
  - Termination: height < 0.8 or |ang| > 1.0 (handled inside custom_env.step)
  - eps=0.05 (Zhang et al. L-inf attack budget for Walker2D)
"""
import sys
import os
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPT_ROOT  = os.path.join(_REPO_ROOT, "other_attacks", "optimal_attack")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _OPT_ROOT not in sys.path:
    sys.path.append(_OPT_ROOT)

from other_attacks.optimal_attack.opt_pg.models import CtsPolicy  # noqa: E402


# ── gym 0.26 compatibility ─────────────────────────────────────────────────────

def _patch_gym_env(gym_env):
    _orig = gym_env.step
    def _step_compat(action):
        r = _orig(action)
        if len(r) == 5:
            o, rew, term, trunc, info = r
            return o, rew, bool(term or trunc), info
        return r
    gym_env.step = _step_compat
    if not hasattr(gym_env, "seed"):
        gym_env.seed = lambda s=None: None
    return gym_env


# ── Helpers ───────────────────────────────────────────────────────────────────

def _raw_obs_from_sim(custom_env) -> np.ndarray:
    """
    Raw 17D Walker2D observation: qpos[1:9] + clip(qvel[:9], -10, 10).
    Walker2D qpos has 9 elements (qpos[0] is the x position, excluded).
    """
    inner = custom_env.env.unwrapped
    qpos  = np.array(inner.sim.data.qpos.flat[1:9])              # 8D
    qvel  = np.clip(inner.sim.data.qvel.flat[:9], -10.0, 10.0)   # 9D
    return np.concatenate([qpos, qvel]).astype(np.float32)


# ── Policy wrappers ───────────────────────────────────────────────────────────

class Walker2DPerfPolicy:
    """
    Native PPO policy for Walker2D. Loaded from the attack checkpoint so
    the policy, ZFilter, and adversary share consistent statistics.
    """

    def __init__(self, model: CtsPolicy, custom_env, attack_model=None,
                 eps: float = 0.05):
        self.model        = model
        self.custom_env   = custom_env
        self.attack_model = attack_model
        self.eps          = eps

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        from copy import deepcopy
        f = getattr(self.custom_env, "new_filter", self.custom_env.state_filter)
        return deepcopy(f)(raw_obs.astype(np.float32)).astype(np.float32)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t   = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            pds = self.model(t)
            act = torch.clamp(pds[0], -1.0, 1.0)
        return act.numpy()[0].astype(np.float32)

    def start_episode(self, max_steps: int = 1000) -> np.ndarray:
        self._step_count = 0
        self._max_steps  = max_steps

        raw = self.custom_env.env.reset()
        if isinstance(raw, tuple):
            raw = raw[0]

        inner  = self.custom_env.env.unwrapped
        qpos   = inner.sim.data.qpos.copy()   # shape (9,)
        qvel   = inner.sim.data.qvel.copy()   # shape (9,)
        uState = np.concatenate([qpos, qvel]).astype(np.float32)  # 18D

        obs_norm = self.custom_env.reset(uState, None, name="Walker2D")
        return obs_norm.astype(np.float32)

    def step(self, action: np.ndarray):
        result, norm_rew, is_done, info = self.custom_env.step(
            action, change_filter=False, name="Walker2D"
        )
        obs_norm = result[1].astype(np.float32)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            is_done = True
        return obs_norm, norm_rew, is_done, info

    @classmethod
    def load(cls, model_path: str, attack_path: str = None,
             naive_policy: bool = False):
        src = attack_path if attack_path is not None else model_path
        ck  = torch.load(src, map_location="cpu")

        if naive_policy and attack_path is not None:
            ck_naive  = torch.load(model_path, map_location="cpu")
            policy_ck = ck_naive
            env_ck    = ck_naive
        else:
            policy_ck = ck
            env_ck    = ck

        model = CtsPolicy(17, 6, "orthogonal")
        model.load_state_dict(policy_ck["policy_model"])
        model.log_stdev.data[:] = -100
        model.eval()

        custom_env = env_ck["envs"][0]
        custom_env.normalizer_read_only = True
        _patch_gym_env(custom_env.env)

        attack_model = None
        if attack_path is not None:
            attack_model = CtsPolicy(17, 17, "orthogonal")
            attack_model.load_state_dict(ck["adversary_policy_model"])
            attack_model.log_stdev.data[:] = -100
            attack_model.eval()

        return cls(model, custom_env, attack_model=attack_model)


class Walker2DBackupPolicy:
    """
    ATLA-defended backup policy for Walker2D. Uses its own ZFilter.
    """

    def __init__(self, model: CtsPolicy, custom_env):
        self.model      = model
        self.custom_env = custom_env

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:
        return self.custom_env.state_filter(
            raw_obs.astype(np.float32)
        ).astype(np.float32)

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t   = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            pds = self.model(t)
            act = torch.clamp(pds[0], -1.0, 1.0)
        return act.numpy()[0].astype(np.float32)

    @classmethod
    def load(cls, model_path: str):
        ck = torch.load(model_path, map_location="cpu")

        model = CtsPolicy(17, 6, "orthogonal")
        model.load_state_dict(ck["policy_model"])
        model.log_stdev.data[:] = -100
        model.eval()

        custom_env = ck["envs"][0]
        custom_env.normalizer_read_only = True
        _patch_gym_env(custom_env.env)

        return cls(model, custom_env)
