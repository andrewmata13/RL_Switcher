import random
import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def denormalize_eps(epsilon_norm: np.ndarray, std: np.ndarray) -> np.ndarray:
    return epsilon_norm * std
