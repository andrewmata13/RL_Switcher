"""
Train a gait-compatible robust backup for the clean pipeline.

Warm-starts from a PPO checkpoint and fine-tunes with Gaussian observation
noise augmentation.  The frozen norm stats are inherited from the PPO
checkpoint so the backup shares the same obs normalization — this keeps the
learned gait close to PPO's, making ATLA→PPO transitions safe.

Key idea:
  The standard ATLA backup diverges from PPO's gait because it was trained
  from scratch with a different reward landscape (adversarial training).
  Here we start from PPO and only perturb observations during training,
  so the policy learns to be robust while keeping PPO-like movement patterns.

Usage:
    python3.8 scripts/train_clean_robust_backup.py --env hopper \
        --ppo-path Hopper/Hopper_Clean_PPO.pt \
        --output Hopper/Hopper_Clean_RobustBackup.pt \
        --obs-noise-sigma 0.10 \
        --total-steps 300000 --seed 0

    # Test against L2 adversary:
    python3.8 scripts/train_clean_robust_backup.py --env hopper \
        --ppo-path Hopper/Hopper_Clean_PPO.pt \
        --adv-path Hopper/Hopper_Clean_PPO_Adv.pt \
        --output Hopper/Hopper_Clean_RobustBackup.pt \
        --obs-noise-sigma 0.10 --adv-prob 0.5 --adv-eps 0.13 \
        --total-steps 300000 --seed 0
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.compat import ensure_paths
from rs_switcher_common.attacks import opt_attack

ensure_paths()
from other_attacks.optimal_attack.opt_pg.models import CtsPolicy, ValueDenseNet


_GYM_IDS = {
    "Hopper":   "Hopper-v4",
    "Cheetah":  "HalfCheetah-v4",
    "Walker2D": "Walker2d-v4",
}


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
# Frozen normalizer (from PPO checkpoint)
# ---------------------------------------------------------------------------
class FrozenNorm:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std  = std.astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (self.std + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Rollout collection with obs-noise augmentation
# ---------------------------------------------------------------------------
def collect_rollout(env, policy, value_net, norm, T, obs_dim, action_dim,
                    obs_noise_sigma: float,
                    adv_model=None, adv_prob: float = 0.0, adv_eps: float = 0.13):
    """
    Collect T steps.  obs used for policy/value inference is perturbed:
      - Gaussian noise: N(0, sigma^2 I) added every step
      - (optionally) adversarial perturbation at rate adv_prob

    Crucially, the *environment* still receives clean actions — only the
    observation fed to the policy is perturbed, matching the threat model.
    """
    states  = torch.zeros(T, obs_dim)
    actions = torch.zeros(T, action_dim)
    lps     = torch.zeros(T)
    rewards = torch.zeros(T)
    nd      = torch.zeros(T)
    values  = torch.zeros(T)

    obs_raw, _ = env.reset()
    obs = norm.normalize(obs_raw)
    done = False
    ep_steps = 0

    for t in range(T):
        if done:
            obs_raw, _ = env.reset()
            obs = norm.normalize(obs_raw)
            done = False
            ep_steps = 0

        # Perturb observation for policy input
        obs_perturbed = obs + np.random.normal(0, obs_noise_sigma, obs.shape).astype(np.float32)

        # Optional: mix in adversarial perturbations
        if adv_model is not None and adv_prob > 0 and np.random.random() < adv_prob:
            obs_perturbed = opt_attack(adv_model, obs, eps=adv_eps, norm='l2')

        obs_t = torch.tensor(obs_perturbed, dtype=torch.float32).unsqueeze(0)
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
# Evaluation (clean obs, deterministic)
# ---------------------------------------------------------------------------
def evaluate(env, policy, norm, n_episodes=5):
    policy.eval()
    returns, falls = [], 0
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
        if done and steps < 1000:
            falls += 1
    policy.train()
    return returns, falls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env",              required=True, choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--ppo-path",         required=True,
                   help="PPO checkpoint to warm-start from (provides frozen norm stats)")
    p.add_argument("--adv-path",         default=None,
                   help="Adversary checkpoint for optional adversarial obs mixing")
    p.add_argument("--output",           required=True)
    p.add_argument("--obs-noise-sigma",  type=float, default=0.10,
                   help="Std of Gaussian obs noise added during training")
    p.add_argument("--adv-prob",         type=float, default=0.0,
                   help="Fraction of steps that use adversarial obs instead of Gaussian noise")
    p.add_argument("--adv-eps",          type=float, default=0.13,
                   help="L2 eps for adversarial obs perturbation")
    p.add_argument("--total-steps",      type=int,   default=300_000)
    p.add_argument("--rollout-len",      type=int,   default=2048)
    p.add_argument("--lr",               type=float, default=1e-4,
                   help="Lower than PPO training to preserve gait")
    p.add_argument("--val-lr",           type=float, default=2.5e-4)
    p.add_argument("--clip-eps",         type=float, default=0.2)
    p.add_argument("--ppo-epochs",       type=int,   default=10)
    p.add_argument("--minibatch",        type=int,   default=64)
    p.add_argument("--entropy",          type=float, default=0.005,
                   help="Lower than PPO training to preserve learned behavior")
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--lam",              type=float, default=0.95)
    p.add_argument("--eval-every",       type=int,   default=20)
    p.add_argument("--eval-episodes",    type=int,   default=5)
    p.add_argument("--seed",             type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config  = ENV_REGISTRY[args.env]
    gym_id  = _GYM_IDS[config.name]

    # Load PPO checkpoint for warm-start and frozen norm
    ppo_ck = torch.load(args.ppo_path, map_location="cpu")
    norm = FrozenNorm(ppo_ck["norm_mean"], ppo_ck["norm_std"])
    print(f"Loaded norm from {args.ppo_path}: mean[:4]={norm.mean[:4].round(3)}, std[:4]={norm.std[:4].round(3)}")

    # Policy and value net — warm-start policy from PPO
    policy    = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
    policy.load_state_dict(ppo_ck["policy_model"])
    # Reset log_stdev to allow exploration
    policy.log_stdev.data[:] = -0.5
    value_net = ValueDenseNet(config.obs_dim, init="orthogonal")

    # Load adversary model if provided
    adv_model = None
    if args.adv_path is not None:
        from rs_switcher_common.clean_policies import CleanPerfPolicy
        perf_adv = CleanPerfPolicy.load(config, args.ppo_path, attack_path=args.adv_path)
        adv_model = perf_adv.attack_model
        print(f"Loaded adversary from {args.adv_path}")

    pol_opt = optim.Adam(policy.parameters(),    lr=args.lr,     eps=1e-5)
    val_opt = optim.Adam(value_net.parameters(), lr=args.val_lr, eps=1e-5)

    env      = gym.make(gym_id)
    env_eval = gym.make(gym_id)

    n_iters = args.total_steps // args.rollout_len
    sched_p = optim.lr_scheduler.LambdaLR(pol_opt, lambda i: 1 - i / n_iters)
    sched_v = optim.lr_scheduler.LambdaLR(val_opt, lambda i: 1 - i / n_iters)

    print(f"\nTraining gait-compatible robust backup on {gym_id}")
    print(f"  obs_noise_sigma={args.obs_noise_sigma}  adv_prob={args.adv_prob}  adv_eps={args.adv_eps}")
    print(f"  total_steps={args.total_steps}  lr={args.lr}  entropy={args.entropy}\n")

    policy.train(); value_net.train()

    for it in range(n_iters):
        batch = collect_rollout(env, policy, value_net, norm,
                                args.rollout_len, config.obs_dim, config.action_dim,
                                obs_noise_sigma=args.obs_noise_sigma,
                                adv_model=adv_model, adv_prob=args.adv_prob,
                                adv_eps=args.adv_eps)
        adv_gae, ret = compute_gae(batch["rewards"], batch["values"],
                                   batch["not_dones"], args.gamma, args.lam)
        batch["advantages"] = adv_gae
        batch["returns"]    = ret

        ppo_update(policy, value_net, pol_opt, val_opt, batch,
                   clip_eps=args.clip_eps, epochs=args.ppo_epochs,
                   minibatch=args.minibatch, entropy_coeff=args.entropy)
        sched_p.step(); sched_v.step()

        if (it + 1) % args.eval_every == 0 or it == n_iters - 1:
            rets, falls = evaluate(env_eval, policy, norm, args.eval_episodes)
            steps_done  = (it + 1) * args.rollout_len
            print(f"  iter={it+1:4d}  steps={steps_done:7d}  "
                  f"return={np.mean(rets):.1f}+/-{np.std(rets):.1f}  "
                  f"falls={falls}/{args.eval_episodes}")

    env.close(); env_eval.close()

    # Final deterministic evaluation
    print("\n=== Final evaluation (clean obs, deterministic) ===")
    env_f = gym.make(gym_id)
    rets, falls = evaluate(env_f, policy, norm, n_episodes=20)
    env_f.close()
    print(f"  mean={np.mean(rets):.1f}  std={np.std(rets):.1f}  falls={falls}/20")

    # Save in CleanBackupPolicy format (same as CleanPerfPolicy)
    policy.log_stdev.data[:] = -100
    policy.eval()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "policy_model": policy.state_dict(),
        "norm_mean":    ppo_ck["norm_mean"],
        "norm_std":     ppo_ck["norm_std"],
        "obs_dim":      config.obs_dim,
        "action_dim":   config.action_dim,
        "env_name":     config.name,
    }, args.output)
    print(f"\nSaved robust backup to {args.output}")


if __name__ == "__main__":
    main()
