"""
Adversarial attack for HalfCheetah.

opt_attack() applies the pre-trained Zhang et al. adversary:
    adv_obs = obs + tanh(attack_net(obs)) * eps
where obs is ZFilter-normalized and eps=0.15 (L-inf budget).
"""
import numpy as np
import torch

from other_attacks.optimal_attack.opt_pg.models import CtsPolicy


def opt_attack(attack_model: CtsPolicy, obs_norm: np.ndarray,
               eps: float = 0.15) -> np.ndarray:
    """
    Apply the pre-trained optimal adversary to a normalized observation.

    Parameters
    ----------
    attack_model : CtsPolicy(17, 17)
        Pre-trained adversary network.
    obs_norm : np.ndarray (17,)
        ZFilter-normalized PPO observation.
    eps : float
        L-inf perturbation budget (default 0.15 for HalfCheetah).

    Returns
    -------
    np.ndarray (17,)
        Adversarially perturbed observation (still in normalized space).
    """
    with torch.no_grad():
        t   = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        pds = attack_model(t)
        delta = torch.tanh(attack_model.sample(pds))
    return (obs_norm + delta.numpy()[0] * eps).astype(np.float32)
