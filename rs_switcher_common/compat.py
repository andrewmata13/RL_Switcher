"""Shared compatibility utilities for MuJoCo env wrappers."""
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPT_ROOT = os.path.join(_REPO_ROOT, "other_attacks", "optimal_attack")


def ensure_paths():
    """Ensure policy_gradients/ and opt_pg/ are importable.

    policy_gradients/ must come first so that torch.load() unpickles
    envs[0] using the correct Env class (which has reset/step with name=).
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    if _OPT_ROOT not in sys.path:
        sys.path.append(_OPT_ROOT)


def patch_gym_env(gym_env):
    """Patch the inner gym TimeLimit wrapper for gym 0.26.

    custom_env.step() calls self.env.step() expecting a 4-tuple.
    gym 0.26 returns a 5-tuple.
    """
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
