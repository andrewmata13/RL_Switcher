from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class EnvConfig:
    """Environment-specific parameters for MuJoCo envs."""
    name: str                           # custom_env API name ("Hopper", "Cheetah", "Walker2D")
    obs_dim: int                        # dimensionality of normalized observation
    action_dim: int                     # dimensionality of action space
    eps: float                          # L-inf attack budget
    qpos_slice: Tuple[int, int]         # (start, stop) for qpos in _raw_obs_from_sim
    qvel_slice: Tuple[int, int]         # (start, stop) for qvel in _raw_obs_from_sim
    qvel_clip: Optional[float] = None   # if set, clip qvel to [-clip, clip]


HOPPER = EnvConfig(
    name="Hopper", obs_dim=11, action_dim=3, eps=0.075,
    qpos_slice=(1, 6), qvel_slice=(0, 6), qvel_clip=10.0,
)

HALFCHEETAH = EnvConfig(
    name="Cheetah", obs_dim=17, action_dim=6, eps=0.15,
    qpos_slice=(1, 9), qvel_slice=(0, 9), qvel_clip=None,
)

WALKER2D = EnvConfig(
    name="Walker2D", obs_dim=17, action_dim=6, eps=0.05,
    qpos_slice=(1, 9), qvel_slice=(0, 9), qvel_clip=10.0,
)

ENV_REGISTRY = {
    "hopper": HOPPER,
    "halfcheetah": HALFCHEETAH,
    "walker2d": WALKER2D,
}
