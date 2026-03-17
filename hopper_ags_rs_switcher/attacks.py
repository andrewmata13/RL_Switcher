"""
Adversarial attack for Hopper using the pre-trained optimal attack network.

The attack is Zhang et al.'s optimal adversary: a CtsPolicy network that maps
the current normalized observation to a perturbation vector.  The actual
perturbation applied is:

    adv_obs = obs + tanh(attack_network(obs)) * eps

where obs and adv_obs are both in NORMALIZED observation space (i.e. after
the ZFilter has been applied).  eps=0.075 matches the training budget.

This replaces the PGD-based attack used for CartPole.
"""
import numpy as np
import torch


def opt_attack(attack_model, obs_norm: np.ndarray, eps: float = 0.075) -> np.ndarray:
    """
    Apply the pre-trained adversarial attack network to a normalized observation.

    Parameters
    ----------
    attack_model : CtsPolicy(11, 11)
        Pre-trained adversary network loaded from checkpoint.
    obs_norm : np.ndarray, shape (11,)
        Normalized observation (output of ZFilter).
    eps : float
        Perturbation budget (default 0.075, same as training).

    Returns
    -------
    adv_obs_norm : np.ndarray, shape (11,)
        Adversarially perturbed normalized observation.
    """
    with torch.no_grad():
        t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        pds = attack_model(t)
        perturbation = attack_model.sample(pds)
        adv_obs = obs_norm + torch.tanh(perturbation).numpy()[0] * eps
    return adv_obs.astype(np.float32)
