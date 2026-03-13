import numpy as np
import torch
from .controllers import PerfPolicy
from .utils import normalize


def pgd_l2_attack(
    perf_policy: PerfPolicy,
    obs: np.ndarray,
    epsilon_l2: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    n_steps: int = 10,
    alpha: float = None,
    n_restarts: int = 1,
) -> np.ndarray:
    """PGD L2 attack in normalized observation space.

    epsilon_l2 is the L2 budget in normalized space.
    Returns a perturbed observation in raw space.
    """
    if alpha is None:
        alpha = min(0.25, 2.5 * epsilon_l2 / n_steps)

    mean_t = torch.tensor(state_mean.astype(np.float32), device=perf_policy.device)
    std_t = torch.tensor(state_std.astype(np.float32), device=perf_policy.device)
    obs_norm = normalize(obs, state_mean, state_std).astype(np.float32)

    def _margin(x_norm: np.ndarray) -> float:
        x_raw_t = torch.tensor(x_norm, dtype=torch.float32, device=perf_policy.device) * std_t + mean_t
        with torch.no_grad():
            dist = perf_policy.policy.get_distribution(x_raw_t.unsqueeze(0))
            logits = dist.distribution.logits.squeeze(0)
            top2 = torch.topk(logits, k=2)
        return float(top2.values[0] - top2.values[1])

    def _run_pgd(norm_start: np.ndarray) -> np.ndarray:
        adv_norm = norm_start.copy()
        for _ in range(n_steps):
            adv_norm_t = torch.tensor(adv_norm, dtype=torch.float32,
                                      device=perf_policy.device, requires_grad=True)
            obs_raw_t = adv_norm_t * std_t + mean_t
            dist = perf_policy.policy.get_distribution(obs_raw_t.unsqueeze(0))
            logits = dist.distribution.logits.squeeze(0)
            top2 = torch.topk(logits, k=2)
            margin = top2.values[0] - top2.values[1]
            perf_policy.policy.zero_grad(set_to_none=True)
            margin.backward()

            grad = adv_norm_t.grad.detach().cpu().numpy()
            grad_norm = np.linalg.norm(grad)
            grad_unit = grad / grad_norm if grad_norm > 0 else grad

            adv_norm = adv_norm - alpha * grad_unit
            delta = adv_norm - obs_norm
            delta_norm = np.linalg.norm(delta)
            if delta_norm > epsilon_l2:
                delta = delta * (epsilon_l2 / delta_norm)
            adv_norm = obs_norm + delta
        return adv_norm

    best_adv_norm = obs_norm.copy()
    best_margin = _margin(obs_norm)

    for restart in range(n_restarts):
        if restart == 0:
            start = obs_norm.copy()
        else:
            noise = np.random.randn(*obs_norm.shape).astype(np.float32)
            noise /= np.linalg.norm(noise) + 1e-12
            noise *= epsilon_l2 * (np.random.uniform() ** (1.0 / obs_norm.shape[0]))
            start = obs_norm + noise

        adv_norm = _run_pgd(start)
        m = _margin(adv_norm)
        if m < best_margin:
            best_margin = m
            best_adv_norm = adv_norm

    # Return in raw observation space
    return (best_adv_norm * state_std + state_mean).astype(np.float32)
