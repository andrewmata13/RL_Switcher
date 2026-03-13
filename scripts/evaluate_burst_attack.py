import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from cartpole_ags_rs_switcher.attacks import pgd_l2_attack
from cartpole_ags_rs_switcher.controllers import PerfPolicy, QuantizedLQRBackup
from cartpole_ags_rs_switcher.evaluation import (
    AlwaysPerfController, AlwaysBackupController,
    UncertifiedSwitcherController, CertifiedSwitcherController,
)
from cartpole_ags_rs_switcher.models import SwitcherMLP
from cartpole_ags_rs_switcher.rs import VanillaRSSwitcher


# ==========================================================
# Attack-aware rollout evaluation
# ==========================================================

@dataclass
class AttackConfig:
    epsilon_l2: float          # L2 budget in normalized observation space
    state_mean: np.ndarray = None
    state_std: np.ndarray = None
    burst_k: int = 5
    horizon_limit: int = 500
    attack_mode: str = "none"  # one of {none, fixed, oracle}
    burst_start: int = 25      # used when attack_mode == fixed


def rollout_episode(
    env_id: str,
    controller,
    perf_policy: PerfPolicy,
    episode_seed: int,
    attack_cfg: AttackConfig,
    attack_start_override: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=episode_seed)
    done = False
    total_reward = 0.0
    t = 0
    infos: List[Dict[str, float]] = []

    if attack_cfg.attack_mode == "none":
        attack_start = None
    elif attack_cfg.attack_mode == "fixed":
        attack_start = attack_cfg.burst_start if attack_start_override is None else attack_start_override
    elif attack_cfg.attack_mode == "oracle":
        attack_start = attack_start_override
    else:
        raise ValueError(f"Unknown attack_mode={attack_cfg.attack_mode}")

    while not done and t < attack_cfg.horizon_limit:
        obs_for_controller = np.array(obs, dtype=np.float32).copy()

        if attack_start is not None and attack_start <= t < attack_start + attack_cfg.burst_k:
            obs_for_controller = pgd_l2_attack(perf_policy, obs_for_controller, attack_cfg.epsilon_l2,
                                               attack_cfg.state_mean, attack_cfg.state_std)

        action, info = controller.select(obs_for_controller)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        infos.append(info)
        done = terminated or truncated
        t += 1

    env.close()

    agg = {
        "return": total_reward,
        "allow_perf_mean": float(np.mean([x["allow_perf"] for x in infos])) if infos else 0.0,
        "p_critical_mean": float(np.nanmean([x["p_critical"] for x in infos])) if infos else np.nan,
        "p_allow_mean": float(np.nanmean([x["p_allow"] for x in infos])) if infos else np.nan,
        "R_rs_mean": float(np.nanmean([x["R_rs"] for x in infos])) if infos else np.nan,
        "R_exec_mean": float(np.nanmean([x["R_exec"] for x in infos])) if infos else np.nan,
        "episode_len": len(infos),
        "attack_start": -1 if attack_start is None else int(attack_start),
    }
    return total_reward, agg


def evaluate_controller_under_attack(
    env_id: str,
    controller,
    perf_policy: PerfPolicy,
    episodes: int,
    seed: int,
    attack_cfg: AttackConfig,
) -> Dict[str, float]:
    returns: List[float] = []
    allow_means: List[float] = []
    pcrit_means: List[float] = []
    pallow_means: List[float] = []
    Rrs_means: List[float] = []
    Rexec_means: List[float] = []
    chosen_starts: List[int] = []

    for ep in range(episodes):
        episode_seed = seed + ep

        if attack_cfg.attack_mode in ("none", "fixed"):
            ret, agg = rollout_episode(env_id, controller, perf_policy, episode_seed, attack_cfg)
        else:
            candidate_starts = list(range(0, attack_cfg.horizon_limit - attack_cfg.burst_k + 1, max(1, attack_cfg.burst_k)))
            best_ret = None
            best_agg = None
            for start in candidate_starts:
                ret_i, agg_i = rollout_episode(env_id, controller, perf_policy, episode_seed, attack_cfg, attack_start_override=start)
                if best_ret is None or ret_i < best_ret:
                    best_ret = ret_i
                    best_agg = agg_i
            ret, agg = best_ret, best_agg

        returns.append(float(ret))
        allow_means.append(agg["allow_perf_mean"])
        pcrit_means.append(agg["p_critical_mean"])
        pallow_means.append(agg["p_allow_mean"])
        Rrs_means.append(agg["R_rs_mean"])
        Rexec_means.append(agg["R_exec_mean"])
        chosen_starts.append(int(agg["attack_start"]))

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_allow_perf": float(np.mean(allow_means)),
        "mean_p_critical": float(np.nanmean(pcrit_means)),
        "mean_p_allow": float(np.nanmean(pallow_means)),
        "mean_R_rs": float(np.nanmean(Rrs_means)),
        "mean_R_exec": float(np.nanmean(Rexec_means)),
        "attack_start_mean": float(np.mean(chosen_starts)) if chosen_starts else -1.0,
    }


# ==========================================================
# Script entry point
# ==========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CartPole controllers under burst observation attacks.")
    parser.add_argument("--perf-path", type=str, required=True)
    parser.add_argument("--switcher-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--n-samples", type=int, default=1000, help="MC samples for RS certification.")
    parser.add_argument("--confidence", type=float, default=0.001, help="Clopper-Pearson confidence level.")
    parser.add_argument("--delta-budget-l2", type=float, required=True)
    parser.add_argument("--epsilon-l2", type=float, default=0.5, help="L2 radius of PGD attack in normalized observation space.")
    parser.add_argument("--burst-k", type=int, default=20)
    parser.add_argument("--attack-mode", type=str, default="fixed", choices=["none", "fixed", "oracle"])
    parser.add_argument("--burst-start", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perf = PerfPolicy.load(args.perf_path, device=device)
    backup = QuantizedLQRBackup()

    data = np.load(args.dataset)
    mean = data["state_mean"]
    std = data["state_std"]

    ckpt = torch.load(args.switcher_path, map_location="cpu")
    switcher = SwitcherMLP(obs_dim=int(ckpt["obs_dim"]), hidden_dim=int(ckpt["hidden_dim"]))
    switcher.load_state_dict(ckpt["state_dict"])
    switcher.eval()

    rs = VanillaRSSwitcher(switcher, mean, std, sigma=float(args.sigma),
                           n_samples=args.n_samples, confidence=args.confidence)

    attack_cfg = AttackConfig(
        epsilon_l2=float(args.epsilon_l2),
        state_mean=mean,
        state_std=std,
        burst_k=int(args.burst_k),
        horizon_limit=500,
        attack_mode=args.attack_mode,
        burst_start=int(args.burst_start),
    )

    controllers = {
        "always_perf": AlwaysPerfController(perf),
        "always_backup": AlwaysBackupController(backup),
        "uncertified_switcher": UncertifiedSwitcherController(perf, backup, rs),
        "certified_switcher": CertifiedSwitcherController(perf, backup, rs, delta_budget_l2=float(args.delta_budget_l2)),
    }

    print("=== Burst-attack evaluation ===")
    print(f"env={args.env_id}")
    print(f"episodes={args.episodes} seed={args.seed}")
    print(f"attack_mode={args.attack_mode} burst_k={args.burst_k} burst_start={args.burst_start}")
    print(f"epsilon_l2={args.epsilon_l2} delta_budget_l2={args.delta_budget_l2} sigma={args.sigma}")
    print(f"n_samples={args.n_samples} confidence={args.confidence}")
    print()

    for name, controller in controllers.items():
        metrics = evaluate_controller_under_attack(
            env_id=args.env_id,
            controller=controller,
            perf_policy=perf,
            episodes=args.episodes,
            seed=args.seed,
            attack_cfg=attack_cfg,
        )
        print(f"[{name}]")
        print(f"  mean return      = {metrics['mean_return']:.2f}")
        print(f"  std return       = {metrics['std_return']:.2f}")
        print(f"  mean allow_perf  = {metrics['mean_allow_perf']:.3f}")
        print(f"  mean p_critical  = {metrics['mean_p_critical']:.3f}")
        print(f"  mean R_rs        = {metrics['mean_R_rs']:.4f}")
        print(f"  mean R_exec      = {metrics['mean_R_exec']:.4f}")
        print(f"  mean attack_start= {metrics['attack_start_mean']:.1f}")
        print()


if __name__ == "__main__":
    main()
