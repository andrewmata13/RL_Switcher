"""
Train adversary and/or policy with L2 norm constraint via PPO.

Two modes:
  --mode adv-only    Train L2 adversary against FROZEN PPO policy.
  --mode minimax     Co-train policy (ATLA) and L2 adversary (minimax).

The trained models are saved in the same checkpoint format used by the
existing pipeline (MuJoCoPerfPolicy.load / MuJoCoBackupPolicy.load).

Usage — adversary only (step 1):
    python3.8 scripts/train_l2_adversary.py --mode adv-only --env hopper \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --eps 0.5 --train-steps 500 --seed 0 \
        --output Hopper/Hopper_Attack_PPO_L2.model

Usage — ATLA minimax from scratch (step 2a):
    python3.8 scripts/train_l2_adversary.py --mode minimax --env hopper \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --eps 0.5 --train-steps 1000 --seed 0 \
        --output Hopper/Hopper_ATLA_L2.model

Usage — ATLA minimax warm-started from existing ATLA (step 2b, faster):
    python3.8 scripts/train_l2_adversary.py --mode minimax --env hopper \
        --attack-path Hopper/Hopper_Attack_PPO.model \
        --warm-start Hopper/Hopper_ATLA.model \
        --eps 0.5 --train-steps 500 --seed 0 \
        --output Hopper/Hopper_ATLA_L2.model
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle, io

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.compat import ensure_paths, patch_gym_env

ensure_paths()
from other_attacks.optimal_attack.opt_pg.models import CtsPolicy, ValueDenseNet


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
# L2 perturbation (numpy, for trajectory collection)
# ---------------------------------------------------------------------------
def l2_perturb(raw_output_np, eps):
    """tanh squash + L2 ball projection."""
    delta = np.tanh(raw_output_np)
    n = np.linalg.norm(delta)
    if n > 1e-8:
        delta = delta / n * eps
    return delta.astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory collection — adversary-only (frozen PPO)
# ---------------------------------------------------------------------------
def collect_adv_only(perf, adv_policy, adv_value, eps, T, config):
    """Adversary attacks frozen PPO every step.  Returns adversary trajectory."""
    D = config.obs_dim
    states  = torch.zeros(T, D)
    actions = torch.zeros(T, D)
    lps     = torch.zeros(T)
    rewards = torch.zeros(T)
    nd      = torch.zeros(T)
    values  = torch.zeros(T)

    obs = perf.start_episode()
    prev_total = 0.0
    done = False

    for t in range(T):
        if done:
            obs = perf.start_episode()
            prev_total = 0.0
            done = False

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        states[t] = obs_t.squeeze(0)

        with torch.no_grad():
            values[t] = adv_value(obs_t).squeeze()

        mean, std = adv_policy(obs_t)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.sample()
        lps[t]     = dist.log_prob(raw).sum(-1).squeeze(0).detach()
        actions[t] = raw.squeeze(0).detach()

        delta = l2_perturb(raw.squeeze(0).detach().numpy(), eps)
        adv_obs = (obs + delta).astype(np.float32)

        action_ppo = perf.predict(adv_obs)
        obs, _, done, _ = perf.step(action_ppo)

        cur_total = perf.custom_env.total_true_reward
        rewards[t] = -(cur_total - prev_total)   # negate for adversary
        prev_total = cur_total
        nd[t] = 0.0 if done else 1.0

    return dict(states=states, actions=actions, log_probs=lps,
                rewards=rewards, not_dones=nd, values=values)


# ---------------------------------------------------------------------------
# Trajectory collection — minimax (trainable policy + adversary)
# ---------------------------------------------------------------------------
def collect_minimax(policy, adv_policy, policy_value, adv_value,
                    custom_env, eps, T, config, for_adversary=False):
    """
    Collect T steps for minimax training.

    The policy acts on perturbed observations (like ATLA seeing attacked obs
    during training).  The adversary perturbs every step.

    for_adversary=True:  return adversary trajectory (rewards negated)
    for_adversary=False: return policy trajectory (rewards positive)
    """
    D = config.obs_dim
    A = config.action_dim
    act_dim = D if for_adversary else A

    states  = torch.zeros(T, D)
    actions = torch.zeros(T, act_dim)
    lps     = torch.zeros(T)
    rewards = torch.zeros(T)
    nd      = torch.zeros(T)
    values  = torch.zeros(T)

    # Reset env
    inner = custom_env.env.unwrapped
    raw = custom_env.env.reset()
    if isinstance(raw, tuple):
        raw = raw[0]
    qpos = inner.sim.data.qpos.copy()
    qvel = inner.sim.data.qvel.copy()
    uState = np.concatenate([qpos, qvel]).astype(np.float32)
    obs = custom_env.reset(uState, None, name=config.name).astype(np.float32)

    prev_total = 0.0
    done = False
    step_count = 0

    for t in range(T):
        if done:
            raw = custom_env.env.reset()
            if isinstance(raw, tuple):
                raw = raw[0]
            qpos = inner.sim.data.qpos.copy()
            qvel = inner.sim.data.qvel.copy()
            uState = np.concatenate([qpos, qvel]).astype(np.float32)
            obs = custom_env.reset(uState, None, name=config.name).astype(np.float32)
            prev_total = 0.0
            step_count = 0
            done = False

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        states[t] = obs_t.squeeze(0)

        # --- adversary perturbation ---
        with torch.no_grad() if not for_adversary else torch.enable_grad():
            pass  # just context placeholder

        adv_mean, adv_std = adv_policy(obs_t)
        adv_dist = torch.distributions.Normal(adv_mean, adv_std)
        adv_raw = adv_dist.sample()

        delta = l2_perturb(adv_raw.squeeze(0).detach().numpy(), eps)
        perturbed_obs = (obs + delta).astype(np.float32)

        # --- policy action ---
        pol_obs_t = torch.tensor(perturbed_obs, dtype=torch.float32).unsqueeze(0)
        pol_mean, pol_std = policy(pol_obs_t)
        pol_dist = torch.distributions.Normal(pol_mean, pol_std)
        pol_raw = pol_dist.sample()
        pol_action = torch.clamp(pol_raw, -1.0, 1.0).squeeze(0).detach().numpy()

        # Record trajectory for whichever agent we're collecting for
        if for_adversary:
            with torch.no_grad():
                values[t] = adv_value(obs_t).squeeze()
            lps[t]     = adv_dist.log_prob(adv_raw).sum(-1).squeeze(0).detach()
            actions[t] = adv_raw.squeeze(0).detach()
        else:
            with torch.no_grad():
                values[t] = policy_value(pol_obs_t).squeeze()
            lps[t]     = pol_dist.log_prob(pol_raw).sum(-1).squeeze(0).detach()
            actions[t] = pol_raw.squeeze(0).detach()

        # --- step environment ---
        result, norm_rew, is_done, info = custom_env.step(
            pol_action, change_filter=False, name=config.name)
        obs = result[1].astype(np.float32)
        step_count += 1
        if step_count >= 1000:
            is_done = True

        cur_total = custom_env.total_true_reward
        true_rew = cur_total - prev_total
        prev_total = cur_total

        rewards[t] = -true_rew if for_adversary else true_rew
        nd[t] = 0.0 if is_done else 1.0
        done = is_done

    return dict(states=states, actions=actions, log_probs=lps,
                rewards=rewards, not_dones=nd, values=values)


# ---------------------------------------------------------------------------
# PPO update (shared by both agents)
# ---------------------------------------------------------------------------
def ppo_update(policy_net, value_net, pol_opt, val_opt,
               batch, clip_eps=0.2, epochs=10, minibatch_size=64,
               entropy_coeff=0.0, max_grad_norm=0.5):
    states     = batch["states"]
    actions    = batch["actions"]
    old_lps    = batch["log_probs"]
    advantages = batch["advantages"]
    returns    = batch["returns"]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    T = states.shape[0]
    idx = np.arange(T)

    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, T, minibatch_size):
            mb = idx[s:s + minibatch_size]
            mb_s, mb_a, mb_olp = states[mb], actions[mb], old_lps[mb]
            mb_adv, mb_ret = advantages[mb], returns[mb]

            mean, std = policy_net(mb_s)
            dist = torch.distributions.Normal(mean, std)
            new_lp = dist.log_prob(mb_a).sum(-1)
            ent = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_lp - mb_olp)
            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            pol_loss = -torch.min(s1, s2).mean() - entropy_coeff * ent

            pol_opt.zero_grad()
            pol_loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
            pol_opt.step()

            v = value_net(mb_s).squeeze(-1)
            val_loss = (v - mb_ret).pow(2).mean()
            val_opt.zero_grad()
            val_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            val_opt.step()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def eval_under_l2_attack(perf, adv_policy, eps, n_episodes=5):
    """Eval PPO return under deterministic L2 adversary."""
    adv_policy.eval()
    rets = []
    for _ in range(n_episodes):
        obs = perf.start_episode(); done = False
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean, _ = adv_policy(obs_t)
                delta = l2_perturb(mean.squeeze(0).numpy(), eps)
            action = perf.predict((obs + delta).astype(np.float32))
            obs, _, done, _ = perf.step(action)
        rets.append(perf.custom_env.total_true_reward)
    adv_policy.train()
    return rets


def eval_minimax_policy(policy, custom_env, config, n_episodes=5):
    """Eval a trainable policy (no attack) to check ATLA-like return."""
    policy.eval()
    inner = custom_env.env.unwrapped
    rets = []
    for _ in range(n_episodes):
        raw = custom_env.env.reset()
        if isinstance(raw, tuple): raw = raw[0]
        qpos = inner.sim.data.qpos.copy()
        qvel = inner.sim.data.qvel.copy()
        uState = np.concatenate([qpos, qvel]).astype(np.float32)
        obs = custom_env.reset(uState, None, name=config.name).astype(np.float32)
        done = False; steps = 0
        while not done and steps < 1000:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mean, _ = policy(obs_t)
                action = torch.clamp(mean, -1, 1).squeeze(0).numpy()
            result, _, done, _ = custom_env.step(action, change_filter=False,
                                                  name=config.name)
            obs = result[1].astype(np.float32)
            steps += 1
            if steps >= 1000: done = True
        rets.append(custom_env.total_true_reward)
    policy.train()
    return rets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["adv-only", "minimax"])
    p.add_argument("--env",  required=True, choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--attack-path", required=True,
                   help="Existing attack checkpoint (PPO + custom_env)")
    p.add_argument("--eps", type=float, required=True,
                   help="L2 perturbation budget (normalized obs space)")
    p.add_argument("--train-steps",   type=int,   default=500)
    p.add_argument("--rollout-len",   type=int,   default=2048)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--val-lr",        type=float, default=1e-3)
    p.add_argument("--clip-eps",      type=float, default=0.2)
    p.add_argument("--ppo-epochs",    type=int,   default=10)
    p.add_argument("--minibatch-size",type=int,   default=64)
    p.add_argument("--entropy-coeff", type=float, default=0.0)
    p.add_argument("--initial-std",   type=float, default=1.0)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--lam",           type=float, default=0.95)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--output",        required=True)
    p.add_argument("--eval-every",    type=int,   default=25)
    p.add_argument("--eval-episodes", type=int,   default=5)
    # Minimax-specific
    p.add_argument("--adv-lr",        type=float, default=3e-5,
                   help="Adversary lr for minimax mode")
    p.add_argument("--adv-val-lr",    type=float, default=3e-5)
    p.add_argument("--policy-steps",  type=int,   default=1,
                   help="Policy update steps per iteration (minimax)")
    p.add_argument("--adv-steps",     type=int,   default=1,
                   help="Adversary update steps per iteration (minimax)")
    p.add_argument("--warm-start",    type=str,   default=None,
                   help="(minimax) Path to existing ATLA checkpoint to warm-start "
                        "policy weights from (much faster convergence)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = ENV_REGISTRY[args.env]
    ck = torch.load(args.attack_path, map_location="cpu")

    # Preserve clean envs for saving
    envs_buf = io.BytesIO()
    pickle.dump(ck["envs"], envs_buf); envs_buf.seek(0)

    custom_env = ck["envs"][0]
    custom_env.normalizer_read_only = True
    patch_gym_env(custom_env.env)

    if args.mode == "adv-only":
        run_adv_only(args, config, ck, custom_env, envs_buf)
    else:
        run_minimax(args, config, ck, custom_env, envs_buf)


# ===================================================================
# MODE 1: adversary-only against frozen PPO
# ===================================================================
def run_adv_only(args, config, ck, custom_env, envs_buf):
    from rs_switcher_common.controllers import MuJoCoPerfPolicy

    ppo = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
    ppo.load_state_dict(ck["policy_model"])
    ppo.log_stdev.data[:] = -100; ppo.eval()
    perf = MuJoCoPerfPolicy(ppo, custom_env, config, attack_model=None)

    adv = CtsPolicy(config.obs_dim, config.obs_dim, "orthogonal")
    if args.initial_std != 1.0:
        adv.log_stdev.data[:] = np.log(args.initial_std)
    adv_val = ValueDenseNet(config.obs_dim, init="orthogonal")

    adv_opt = optim.Adam(adv.parameters(), lr=args.lr, eps=1e-5)
    val_opt = optim.Adam(adv_val.parameters(), lr=args.val_lr, eps=1e-5)
    sched_p = optim.lr_scheduler.LambdaLR(adv_opt, lambda s: 1 - s / args.train_steps)
    sched_v = optim.lr_scheduler.LambdaLR(val_opt, lambda s: 1 - s / args.train_steps)

    print(f"[adv-only] Training L2 adversary for {config.name}")
    print(f"  eps={args.eps}  steps={args.train_steps}  T={args.rollout_len}  lr={args.lr}\n")

    for step in range(args.train_steps):
        batch = collect_adv_only(perf, adv, adv_val, args.eps,
                                 args.rollout_len, config)
        adv_gae, adv_ret = compute_gae(batch["rewards"], batch["values"],
                                        batch["not_dones"], args.gamma, args.lam)
        batch["advantages"] = adv_gae; batch["returns"] = adv_ret

        mean_r = batch["rewards"].mean().item()
        ppo_update(adv, adv_val, adv_opt, val_opt, batch,
                   clip_eps=args.clip_eps, epochs=args.ppo_epochs,
                   minibatch_size=args.minibatch_size,
                   entropy_coeff=args.entropy_coeff)
        sched_p.step(); sched_v.step()

        if step % args.eval_every == 0 or step == args.train_steps - 1:
            rets = eval_under_l2_attack(perf, adv, args.eps, args.eval_episodes)
            print(f"Step {step:4d} | adv_rew {mean_r:+.2f} | "
                  f"PPO under L2: {np.mean(rets):.1f} +/- {np.std(rets):.1f}")

    # Save
    adv.log_stdev.data[:] = -100; adv.eval()
    clean_envs = pickle.load(envs_buf)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "policy_model": ck["policy_model"],
        "adversary_policy_model": adv.state_dict(),
        "envs": clean_envs,
    }, args.output)
    print(f"\nSaved to {args.output}")


# ===================================================================
# MODE 2: minimax ATLA + L2 adversary co-training
# ===================================================================
def run_minimax(args, config, ck, custom_env, envs_buf):
    # Policy (ATLA)
    policy = CtsPolicy(config.obs_dim, config.action_dim, "orthogonal")
    pol_val = ValueDenseNet(config.obs_dim, init="orthogonal")

    if args.warm_start:
        ws_ck = torch.load(args.warm_start, map_location="cpu")
        policy.load_state_dict(ws_ck["policy_model"])
        print(f"  Warm-started policy from {args.warm_start}")
        # Load adversary weights too if present
        if "adversary_policy_model" in ws_ck:
            pass  # will init fresh adversary below; old L-inf one won't help
    else:
        if args.initial_std != 1.0:
            policy.log_stdev.data[:] = np.log(args.initial_std)

    pol_opt = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    pol_val_opt = optim.Adam(pol_val.parameters(), lr=args.val_lr, eps=1e-5)

    # Adversary — always fresh (L2 directions differ from L-inf)
    adv = CtsPolicy(config.obs_dim, config.obs_dim, "orthogonal")
    adv_val = ValueDenseNet(config.obs_dim, init="orthogonal")

    adv_opt = optim.Adam(adv.parameters(), lr=args.adv_lr, eps=1e-5)
    adv_val_opt = optim.Adam(adv_val.parameters(), lr=args.adv_val_lr, eps=1e-5)

    # LR annealing
    mk_sched = lambda opt: optim.lr_scheduler.LambdaLR(
        opt, lambda s: 1 - s / args.train_steps)
    scheds = [mk_sched(o) for o in [pol_opt, pol_val_opt, adv_opt, adv_val_opt]]

    print(f"[minimax] Training ATLA + L2 adversary for {config.name}")
    print(f"  eps={args.eps}  steps={args.train_steps}  T={args.rollout_len}")
    print(f"  policy lr={args.lr}  adv lr={args.adv_lr}")
    print(f"  warm_start={args.warm_start or 'None (from scratch)'}\n")

    for step in range(args.train_steps):
        # --- Adversary update(s) ---
        for _ in range(args.adv_steps):
            abatch = collect_minimax(policy, adv, pol_val, adv_val,
                                     custom_env, args.eps, args.rollout_len,
                                     config, for_adversary=True)
            ag, ar = compute_gae(abatch["rewards"], abatch["values"],
                                  abatch["not_dones"], args.gamma, args.lam)
            abatch["advantages"] = ag; abatch["returns"] = ar
            ppo_update(adv, adv_val, adv_opt, adv_val_opt, abatch,
                       clip_eps=args.clip_eps, epochs=args.ppo_epochs,
                       minibatch_size=args.minibatch_size,
                       entropy_coeff=args.entropy_coeff)

        # --- Policy update(s) ---
        for _ in range(args.policy_steps):
            pbatch = collect_minimax(policy, adv, pol_val, adv_val,
                                     custom_env, args.eps, args.rollout_len,
                                     config, for_adversary=False)
            pg, pr = compute_gae(pbatch["rewards"], pbatch["values"],
                                  pbatch["not_dones"], args.gamma, args.lam)
            pbatch["advantages"] = pg; pbatch["returns"] = pr
            ppo_update(policy, pol_val, pol_opt, pol_val_opt, pbatch,
                       clip_eps=args.clip_eps, epochs=args.ppo_epochs,
                       minibatch_size=args.minibatch_size,
                       entropy_coeff=args.entropy_coeff)

        for s in scheds:
            s.step()

        # --- Logging ---
        if step % args.eval_every == 0 or step == args.train_steps - 1:
            pol_rets = eval_minimax_policy(policy, custom_env, config,
                                            args.eval_episodes)
            adv_rew = abatch["rewards"].mean().item()
            pol_rew = pbatch["rewards"].mean().item()
            print(f"Step {step:4d} | pol_rew {pol_rew:+.2f} adv_rew {adv_rew:+.2f} | "
                  f"ATLA clean: {np.mean(pol_rets):.1f} +/- {np.std(pol_rets):.1f}")

    # Save — ATLA checkpoint format (policy_model + envs)
    policy.log_stdev.data[:] = -100; policy.eval()
    adv.log_stdev.data[:] = -100; adv.eval()
    clean_envs = pickle.load(envs_buf)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Save ATLA policy
    torch.save({
        "policy_model": policy.state_dict(),
        "adversary_policy_model": adv.state_dict(),
        "envs": clean_envs,
    }, args.output)
    print(f"\nSaved ATLA + adversary to {args.output}")

    # Also save attack-only checkpoint (adversary + original PPO)
    atk_path = args.output.replace("ATLA", "Attack_ATLA")
    if atk_path != args.output:
        torch.save({
            "policy_model": policy.state_dict(),
            "adversary_policy_model": adv.state_dict(),
            "envs": clean_envs,
        }, atk_path)
        print(f"Saved attack checkpoint to {atk_path}")


if __name__ == "__main__":
    main()
