"""
Adversarial attack using the pre-trained optimal attack network (Zhang et al.).

The attack maps a normalized observation to a perturbation vector:
    L-inf: adv_obs = obs + tanh(attack_network(obs)) * eps
    L2:    adv_obs = obs + eps * delta / max(||delta||_2, 1)

where obs and adv_obs are both in NORMALIZED observation space (after ZFilter).
"""
import numpy as np
import torch


def opt_attack(attack_model, obs_norm: np.ndarray, eps: float,
               norm: str = "linf") -> np.ndarray:
    """
    Apply the pre-trained adversarial attack network to a normalized observation.

    Parameters
    ----------
    attack_model : CtsPolicy(obs_dim, obs_dim)
        Pre-trained adversary network loaded from checkpoint.
    obs_norm : np.ndarray, shape (obs_dim,)
        Normalized observation (output of ZFilter).
    eps : float
        Perturbation budget (L-inf or L2, depending on norm).
    norm : str
        "linf" for L-inf constraint (tanh * eps per dimension),
        "l2" for L2 constraint (project delta onto L2 ball of radius eps).

    Returns
    -------
    adv_obs_norm : np.ndarray, shape (obs_dim,)
        Adversarially perturbed normalized observation.
    """
    with torch.no_grad():
        t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        pds = attack_model(t)
        delta = torch.tanh(attack_model.sample(pds)).numpy()[0]

    if norm == "linf":
        return (obs_norm + delta * eps).astype(np.float32)
    elif norm == "l2":
        # Project onto L2 ball: scale direction to have ||delta||_2 = eps
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 1e-8:
            delta = delta / delta_norm  # unit direction
        return (obs_norm + delta * eps).astype(np.float32)
    else:
        raise ValueError(f"Unknown norm: {norm}. Use 'linf' or 'l2'.")
