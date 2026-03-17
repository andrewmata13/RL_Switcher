"""
Adversarial attack for Walker2D.

opt_attack() applies the pre-trained Zhang et al. adversary:
    adv_obs = obs + tanh(attack_net(obs)) * eps
where obs is ZFilter-normalized and eps=0.05 (L-inf budget).
"""
import numpy as np
import torch

from other_attacks.optimal_attack.opt_pg.models import CtsPolicy


def opt_attack(attack_model: CtsPolicy, obs_norm: np.ndarray,
               eps: float = 0.05) -> np.ndarray:
    with torch.no_grad():
        t   = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        pds = attack_model(t)
        delta = torch.tanh(attack_model.sample(pds))
    return (obs_norm + delta.numpy()[0] * eps).astype(np.float32)
