from dataclasses import dataclass
import numpy as np


@dataclass
class LabelConfig:
    epsilon_l2: float
    burst_k: int = 3
    horizon_h: int = 50
    reward_drop_threshold: float = 30.0
    seed: int = 123
    pgd_steps: int = 3
    n_attack_starts: int = 3


@dataclass
class SwitcherTrainConfig:
    hidden_dim: int = 32
    epochs: int = 40
    lr: float = 1e-3
    batch_size: int = 128


@dataclass
class EvalConfig:
    sigma: float
    delta_budget_l2: float
    episodes: int = 20
    seed: int = 999
