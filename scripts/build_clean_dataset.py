"""
Build adversarial-detection dataset for the clean (ZFilter-free) pipeline.

Stores NORMALIZED observations (obs_ppo = CleanPerfPolicy.normalize(raw)),
so GPSwitcher._normalize_t(obs_ppo) at inference sees the same distribution
as during training.

Usage:
    python3.8 scripts/build_clean_dataset.py --env hopper \
        --ppo-path   Hopper/Hopper_Clean_PPO.pt \
        --adv-path   Hopper/Hopper_Clean_PPO_Adv.pt \
        --dataset-out data/hopper_clean_dataset.npz \
        --episodes 30 --subsample-every 5
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.clean_policies import CleanPerfPolicy
from rs_switcher_common.attacks import opt_attack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env",          required=True, choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--ppo-path",     required=True)
    p.add_argument("--adv-path",     required=True,
                   help="Checkpoint with adversary_policy_model (may equal ppo-path)")
    p.add_argument("--dataset-out",  required=True)
    p.add_argument("--episodes",     type=int, default=30)
    p.add_argument("--subsample-every", type=int, default=5)
    p.add_argument("--backup-path",  required=True,
                   help="CleanBackupPolicy checkpoint (for ATLA-trajectory clean obs)")
    p.add_argument("--attack-eps",   type=float, default=None,
                   help="Override attack eps (default: config.eps)")
    p.add_argument("--attack-norm",  type=str, default="linf", choices=["linf", "l2"],
                   help="Attack norm for generating adversarial examples (default: linf)")
    p.add_argument("--seed",         type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = ENV_REGISTRY[args.env]
    perf = CleanPerfPolicy.load(config, args.ppo_path, attack_path=args.adv_path)
    eps = args.attack_eps if args.attack_eps is not None else config.eps

    X, y = [], []
    total_steps = 0

    # PPO-controlled clean episodes
    for ep in range(args.episodes):
        obs = perf.start_episode()
        done = False
        t = 0
        while not done:
            if t % args.subsample_every == 0:
                # Clean obs (already normalized by CleanPerfPolicy)
                X.append(obs.copy())
                y.append(0)
                # Adversarial obs — attack is applied in normalized space
                adv = opt_attack(perf.attack_model, obs, eps=eps, norm=args.attack_norm)
                X.append(adv.copy())
                y.append(1)

            action = perf.predict(obs)
            obs, _, done, _ = perf.step(action)
            t += 1
        total_steps += t
        if (ep + 1) % 10 == 0:
            print(f"  PPO ep {ep+1}/{args.episodes}  steps={t}  collected={len(X)} samples")

    from rs_switcher_common.clean_policies import CleanBackupPolicy
    from rs_switcher_common.controllers import raw_obs_from_sim
    backup = CleanBackupPolicy.load(config, args.backup_path)

    # ATLA-controlled episodes — add both clean obs (y=0) and adversarial ATLA obs (y=1).
    # The attack model was trained on PPO obs; applying it to ATLA obs gives perturbed
    # ATLA obs which should also be class 1.  This teaches the switcher that any natural
    # Hopper state is clean (y=0) and any perturbed state is adversarial (y=1).
    for ep in range(args.episodes):
        obs_ppo = perf.start_episode()
        done = False
        t = 0
        while not done:
            if t % args.subsample_every == 0:
                # obs_ppo is what the switcher certifies during ATLA recovery
                X.append(obs_ppo.copy())
                y.append(0)
                # Adversarial version of the ATLA-trajectory obs
                adv = opt_attack(perf.attack_model, obs_ppo, eps=eps, norm=args.attack_norm)
                X.append(adv.copy())
                y.append(1)

            raw = raw_obs_from_sim(perf.custom_env, config)
            obs_atla = backup.normalize(raw)
            action = backup.predict(obs_atla)
            obs_ppo, _, done, _ = perf.step(action)
            t += 1
        total_steps += t
        if (ep + 1) % 10 == 0:
            print(f"  ATLA ep {ep+1}/{args.episodes}  steps={t}  collected={len(X)} samples")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    # state_mean/std are stats of the normalized obs stored in X.
    # GPSwitcher._normalize_t applies (obs_ppo - state_mean) / state_std at inference;
    # since obs_ppo is already normalized and state_mean≈0, state_std≈1, this is ~no-op.
    state_mean = X.mean(axis=0)
    state_std  = X.std(axis=0) + 1e-6

    os.makedirs(os.path.dirname(args.dataset_out) or ".", exist_ok=True)
    np.savez(args.dataset_out, X=X, y=y, state_mean=state_mean, state_std=state_std)

    print(f"\nSaved {args.dataset_out}")
    print(f"  {len(X)} samples, adv fraction={y.mean():.3f}")
    print(f"  state_mean[:4]: {state_mean[:4].round(4)}")
    print(f"  state_std[:4]:  {state_std[:4].round(4)}")
    print(f"  X[0] (clean):   {X[0]}")
    print(f"  X[1] (adv):     {X[1]}")


if __name__ == "__main__":
    main()
