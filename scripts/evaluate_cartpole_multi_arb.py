"""
Evaluate CartPole GP switcher under multi-burst and arbitrary attacks.

Attack modes:
  multi     — n_bursts of burst_k attacked steps with cooldown_k gap
  arbitrary — each step independently attacked with probability p = E/500

Usage:
    # Multi-burst (2x100, cooldown=100)
    python3.8 scripts/evaluate_cartpole_multi_arb.py \
        --perf-path models/perf_cartpole_15000k.zip \
        --gp-switcher-path models/cartpole_15k_gp.pt \
        --dataset data/cartpole_15k_dataset.npz \
        --sigma 0.25 --delta-budget-l2 0.25 \
        --episodes 50 --seed 0 --epsilon-l2 1.0 \
        --attack-mode multi --n-bursts 2 --burst-k 100 --cooldown-k 100 \
        --K-enter 5 --K-exit 5 \
        --output-json results/cartpole_multi2x100.json

    # Arbitrary (E=200 expected attacked steps)
    python3.8 scripts/evaluate_cartpole_multi_arb.py \
        --perf-path models/perf_cartpole_15000k.zip \
        --gp-switcher-path models/cartpole_15k_gp.pt \
        --dataset data/cartpole_15k_dataset.npz \
        --sigma 0.25 --delta-budget-l2 0.25 \
        --episodes 50 --seed 0 --epsilon-l2 1.0 \
        --attack-mode arbitrary --expected-attacked 200 \
        --K-enter 5 --K-exit 5 \
        --output-json results/cartpole_arb200.json
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import torch

from cartpole_rs_switcher.attacks import pgd_l2_attack
from cartpole_rs_switcher.controllers import PerfPolicy, QuantizedLQRBackup
from rs_switcher_common.gp_models import load_gp_switcher, GPSwitcher

HORIZON = 500


class AlwaysPerfController:
    def __init__(self, perf):
        self.perf = perf

    def reset(self):
        pass

    def select(self, obs):
        return self.perf.predict(obs), {"allow_perf": 1.0}


class AlwaysBackupController:
    def __init__(self, backup):
        self.backup = backup

    def reset(self):
        pass

    def select(self, obs):
        return self.backup.predict(obs), {"allow_perf": 0.0}


class ContinuousGPController:
    """Hysteresis-based continuous switcher (same design as HalfCheetah)."""

    def __init__(self, perf, backup, gp_switcher, delta, K_enter, K_exit):
        self.perf = perf
        self.backup = backup
        self.gp = gp_switcher
        self.delta = delta
        self.K_enter = K_enter
        self.K_exit = K_exit
        self.reset()

    def reset(self):
        self.in_backup = False
        self.alarm_count = 0
        self.safe_count = 0

    def select(self, obs):
        pred, pA, R = self.gp.certify(obs)
        certified_safe = (pred == 0) and (R >= self.delta)

        if not self.in_backup:
            if not certified_safe:
                self.alarm_count += 1
            else:
                self.alarm_count = max(0, self.alarm_count - 1)
            if self.alarm_count >= self.K_enter:
                self.in_backup = True
                self.alarm_count = 0
                self.safe_count = 0
        else:
            if certified_safe:
                self.safe_count += 1
            else:
                self.safe_count = 0
            if self.safe_count >= self.K_exit:
                self.in_backup = False
                self.alarm_count = 0
                self.safe_count = 0

        allow_perf = not self.in_backup
        action = self.perf.predict(obs) if allow_perf else self.backup.predict(obs)
        return action, {"allow_perf": float(allow_perf)}


def build_attack_mask(attack_mode, n_bursts, burst_k, cooldown_k, expected_attacked, rng):
    """Returns a boolean array of length HORIZON: True = step is attacked."""
    mask = np.zeros(HORIZON, dtype=bool)

    if attack_mode == "multi":
        # Place bursts with cooldown gaps, randomise starting offset
        total_span = n_bursts * burst_k + (n_bursts - 1) * cooldown_k
        max_start = max(0, HORIZON - total_span)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        t = start
        for _ in range(n_bursts):
            mask[t:t + burst_k] = True
            t += burst_k + cooldown_k

    elif attack_mode == "arbitrary":
        p = min(1.0, expected_attacked / HORIZON)
        mask = rng.random(HORIZON) < p

    return mask


def rollout(env_id, controller, perf_policy, episode_seed, attack_mask,
            epsilon_l2, state_mean, state_std):
    import gymnasium as gym
    env = gym.make(env_id)
    obs, _ = env.reset(seed=episode_seed)
    controller.reset()
    total_reward = 0.0
    allow_perfs = []

    for t in range(HORIZON):
        obs_ctrl = np.array(obs, dtype=np.float32)
        if attack_mask[t]:
            obs_ctrl = pgd_l2_attack(perf_policy, obs_ctrl, epsilon_l2, state_mean, state_std)

        action, info = controller.select(obs_ctrl)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        allow_perfs.append(info["allow_perf"])
        if terminated or truncated:
            break

    env.close()
    return total_reward, float(np.mean(allow_perfs))


def evaluate(env_id, controller, perf_policy, episodes, seed,
             attack_mode, n_bursts, burst_k, cooldown_k, expected_attacked,
             epsilon_l2, state_mean, state_std):
    rng = np.random.default_rng(seed)
    returns = []
    allow_perfs = []
    for ep in range(episodes):
        mask = build_attack_mask(attack_mode, n_bursts, burst_k, cooldown_k,
                                 expected_attacked, rng)
        ret, ap = rollout(env_id, controller, perf_policy, seed + ep, mask,
                         epsilon_l2, state_mean, state_std)
        returns.append(float(ret))
        allow_perfs.append(float(ap))
    return returns, allow_perfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-path", required=True)
    parser.add_argument("--gp-switcher-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--delta-budget-l2", type=float, default=0.25)
    parser.add_argument("--epsilon-l2", type=float, default=1.0)
    parser.add_argument("--attack-mode", required=True, choices=["multi", "arbitrary"])
    parser.add_argument("--n-bursts", type=int, default=2)
    parser.add_argument("--burst-k", type=int, default=100)
    parser.add_argument("--cooldown-k", type=int, default=100)
    parser.add_argument("--expected-attacked", type=int, default=200,
                        help="Expected attacked steps for arbitrary mode")
    parser.add_argument("--K-enter", type=int, default=5)
    parser.add_argument("--K-exit", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    perf = PerfPolicy.load(args.perf_path, device=device)
    backup = QuantizedLQRBackup()

    data = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]

    ckpt = torch.load(args.gp_switcher_path, map_location="cpu")
    gp_switcher = GPSwitcher(load_gp_switcher(ckpt), mean, std,
                             sigma=args.sigma, device="cpu")

    controllers = {
        "always_perf": AlwaysPerfController(perf),
        "always_backup": AlwaysBackupController(backup),
        "continuous_gp": ContinuousGPController(
            perf, backup, gp_switcher,
            delta=args.delta_budget_l2,
            K_enter=args.K_enter,
            K_exit=args.K_exit,
        ),
    }

    print(f"attack_mode={args.attack_mode}  episodes={args.episodes}  seed={args.seed}")
    if args.attack_mode == "multi":
        print(f"n_bursts={args.n_bursts}  burst_k={args.burst_k}  cooldown_k={args.cooldown_k}")
    else:
        print(f"expected_attacked={args.expected_attacked}")
    print(f"eps={args.epsilon_l2}  sigma={args.sigma}  delta={args.delta_budget_l2}")
    print(f"K_enter={args.K_enter}  K_exit={args.K_exit}")
    print()

    all_results = {}
    for name, ctrl in controllers.items():
        returns, allow_perfs = evaluate(
            args.env_id, ctrl, perf, args.episodes, args.seed,
            args.attack_mode, args.n_bursts, args.burst_k, args.cooldown_k,
            args.expected_attacked, args.epsilon_l2, mean, std,
        )
        r = np.array(returns)
        ap = np.mean(allow_perfs)
        print(f"[{name}]  median={np.median(r):.0f}  mean={np.mean(r):.0f}  std={np.std(r):.0f}  allow_perf={ap:.3f}")
        all_results[name] = {"returns": returns, "median": float(np.median(r)),
                             "mean": float(np.mean(r)), "std": float(np.std(r)),
                             "mean_allow_perf": float(ap)}

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"config": vars(args), **all_results}, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
