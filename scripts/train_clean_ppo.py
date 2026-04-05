"""
Train a clean PPO policy for a MuJoCo environment using raw gymnasium.

No ZFilter. Uses a running obs normalizer during training that is frozen
after training and saved with the checkpoint. Both this policy and the
subsequently trained ATLA will share these frozen stats, eliminating all
ZFilter switching instability.

Saves: Hopper/Hopper_Clean_PPO.pt
    {policy_model, norm_mean, norm_std, obs_dim, action_dim}

Usage:
    python3.8 scripts/train_clean_ppo.py --env hopper \
        --output Hopper/Hopper_Clean_PPO.pt \
        --total-steps 1000000 --seed 0
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.compat import ensure_paths

ensure_paths()
from other_attacks.optimal_attack.opt_pg.models import CtsPolicy, ValueDenseNet


_GYM_IDS = {
    "Hopper":   "Hopper-v4",
    "Cheetah":  "HalfCheetah-v4",
    "Walker2D": "Walker2d-v4",
}


# ---------------------------------------------------------------------------
# Running normalizer (Welford online algorithm)
# ---------------------------------------------------------------------------
class RunningNorm:
    """Online mean/variance estimator. Freeze after training."""
    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2   = np.ones(dim,  dtype=np.float64)  # sum of squared deviations

    def update(self, x: np.ndarray):
        self.n += 1
        delta  = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(np.maximum(self.M2 / self.n, 1e-8))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (self.std + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, not_dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] * not_dones[t] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * not_dones[t] * gae
        advantages[t] = gae
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def ppo_update(policy, value_net, pol_opt, val_opt, batch,
               clip_eps=0.2, epochs=10, minibatch=64,
               entropy_coeff=0.01, max_grad_norm=0.5):
    states     = batch["states"]
    actions    = batch["actions"]
    old_lps    = batch["log_probs"]
    advantages = batch["advantages"]
    returns    = batch["returns"]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    idx = np.arange(len(states))

    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(states), minibatch):
            mb = idx[s:s + minibatch]
            mb_s   = states[mb];    mb_a   = actions[mb]
            mb_olp = old_lps[mb];   mb_adv = advantages[mb]
            mb_ret = returns[mb]

            mean, std = policy(mb_s)
            dist   = torch.distributions.Normal(mean, std)
            new_lp = dist.log_prob(mb_a).sum(-1)
            ent    = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_lp - mb_olp)
            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            pol_loss = -torch.min(s1, s2).mean() - entropy_coeff * ent

            pol_opt.zero_grad()
            pol_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            pol_opt.step()

            v = value_net(mb_s).squeeze(-1)
            val_loss = (v - mb_ret).pow(2).mean()
            val_opt.zero_grad()
            val_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            val_opt.step()


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
def collect_rollout(env, policy, value_net, norm, T, obs_dim, action_dim):
    """
    Collect T environment steps. Updates norm in-place.
    Returns batch dict with normalized observations.
    """
    states  = torch.zeros(T, obs_dim)
    actions = torch.zeros(T, action_dim)
    lps     = torch.zeros(T)
    rewards = torch.zeros(T)
    nd      = torch.zeros(T)   # not_done flags
    values  = torch.zeros(T)

    obs_raw, _ = env.reset()
    norm.update(obs_raw)
    obs = norm.normalize(obs_raw)
    done = False
    ep_steps = 0

    for t in range(T):
        if done:
            obs_raw, _ = env.reset()
            norm.update(obs_raw)
            obs = norm.normalize(obs_raw)
            done = False
            ep_steps = 0

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        states[t] = obs_t.squeeze(0)

        with torch.no_grad():
            values[t] = value_net(obs_t).squeeze()
            mean, std  = policy(obs_t)
            dist = torch.distributions.Normal(mean, std)
            raw  = dist.sample()
            lps[t]     = dist.log_prob(raw).sum(-1).squeeze(0)
            actions[t] = raw.squeeze(0)
            action = torch.clamp(raw, -1.0, 1.0).squeeze(0).numpy()

        obs_raw, reward, terminated, truncated, _ = env.step(action)
        norm.update(obs_raw)
        obs = norm.normalize(obs_raw)
        done = bool(terminated or truncated)
        ep_steps += 1
        if ep_steps >= 1000:
            done = True

        rewards[t] = float(reward)
        nd[t] = 0.0 if done else 1.0

    return dict(states=states, actions=actions, log_probs=lps,
                rewards=rewards, not_dones=nd, values=values)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(env, policy, norm, n_episodes=5):
    policy.eval()
    returns = []
    for _ in range(n_episodes):
        obs_raw, _ = env.reset()
        total = 0.0
        done = False
        steps = 0
        while not done and steps < 1000:
            obs = norm.normalize(obs_raw)
            with torch.no_grad():
                mean, _ = policy(torch.tensor(obs).unsqueeze(0))
                action  = torch.clamp(mean, -1, 1).squeeze(0).numpy()
            obs_raw, r, terminated, truncated, _ = env.step(action)
            total += r
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
    policy.train()
    return returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env",          required=True, choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--output",       required=True)
    p.add_argument("--total-steps",  type=int,   default=1_000_000)
    p.add_argument("--rollout-len",  type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--val-lr",       type=float, default=2.5e-4)
    p.add_argument("--clip-eps",     type=float, default=0.2)
    p.add_argument("--ppo-epochs",   type=int,   default=10)
    p.add_argument("--minibatch",    type=int,   default=64)
    p.add_argument("--entropy",      type=float, default=0.01)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--lam",          type=float, default=0.95)
    p.add_argument("--eval-every",   type=int,   default=50,
                   help="Evaluate every N PPO iterations")
    p.add_argument("--eval-episodes",type=int,   default=5)
    p.add_argument("--seed",         type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config  = ENV_REGISTRY[args.env]
    gym_id  = _GYM_IDS[config.name]
    env     = gym.make(gym_id)
    env_eval = gym.make(gym_id)

    policy    = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
    value_net = ValueDenseNet(config.obs_dim, init="orthogonal")
    pol_opt   = optim.Adam(policy.parameters(),    lr=args.lr,     eps=1e-5)
    val_opt   = optim.Adam(value_net.parameters(), lr=args.val_lr, eps=1e-5)

    norm = RunningNorm(config.obs_dim)

    n_iters = args.total_steps // args.rollout_len
    # Linear LR annealing (same as Zhang et al.)
    sched_p = optim.lr_scheduler.LambdaLR(pol_opt, lambda i: 1 - i / n_iters)
    sched_v = optim.lr_scheduler.LambdaLR(val_opt, lambda i: 1 - i / n_iters)

    print(f"Training clean PPO on {gym_id}  ({config.obs_dim}D obs, {config.action_dim}D act)")
    print(f"  total_steps={args.total_steps}  rollout_len={args.rollout_len}  "
          f"n_iters={n_iters}  seed={args.seed}")
    print(f"  lr={args.lr}  val_lr={args.val_lr}  entropy={args.entropy}\n")

    policy.train()
    value_net.train()

    for it in range(n_iters):
        batch = collect_rollout(env, policy, value_net, norm,
                                args.rollout_len, config.obs_dim, config.action_dim)
        adv, ret = compute_gae(batch["rewards"], batch["values"],
                               batch["not_dones"], args.gamma, args.lam)
        batch["advantages"] = adv
        batch["returns"]    = ret

        ppo_update(policy, value_net, pol_opt, val_opt, batch,
                   clip_eps=args.clip_eps, epochs=args.ppo_epochs,
                   minibatch=args.minibatch, entropy_coeff=args.entropy)
        sched_p.step()
        sched_v.step()

        if (it + 1) % args.eval_every == 0 or it == n_iters - 1:
            rets = evaluate(env_eval, policy, norm, args.eval_episodes)
            steps_done = (it + 1) * args.rollout_len
            print(f"  iter={it+1:4d}  steps={steps_done:7d}  "
                  f"return={np.mean(rets):.1f}+/-{np.std(rets):.1f}  "
                  f"norm_n={norm.n}")

    env.close()
    env_eval.close()

    # Final evaluation with frozen norm
    print("\n=== Final evaluation (frozen norm) ===")
    env_final = gym.make(gym_id)

    class _FrozenNorm:
        def __init__(self, n): self.mean = n.mean.astype(np.float32); self.std = n.std.astype(np.float32)
        def normalize(self, x): return ((x - self.mean) / (self.std + 1e-8)).astype(np.float32)

    frozen = _FrozenNorm(norm)
    rets = evaluate(env_final, policy, frozen, n_episodes=20)
    env_final.close()
    print(f"  mean={np.mean(rets):.1f}  std={np.std(rets):.1f}  "
          f"min={np.min(rets):.1f}  max={np.max(rets):.1f}")

    # Save
    policy.log_stdev.data[:] = -100
    policy.eval()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "policy_model": policy.state_dict(),
        "norm_mean":    norm.mean.astype(np.float32),
        "norm_std":     norm.std.astype(np.float32),
        "obs_dim":      config.obs_dim,
        "action_dim":   config.action_dim,
        "env_name":     config.name,
    }, args.output)
    print(f"\nSaved clean PPO to {args.output}")


if __name__ == "__main__":
    main()
