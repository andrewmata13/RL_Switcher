"""
Episode trace: single HalfCheetah episode showing per-step:
  - Which policy is active (PPO vs ATLA, colored background stripe)
  - Certified radius R over time
  - Cumulative reward over time
  - Attack window highlighted

Output: poster_figures/episode_trace.pdf
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.controllers import MuJoCoPerfPolicy, MuJoCoBackupPolicy, raw_obs_from_sim
from rs_switcher_common.evaluation import ContinuousSwitcherController, _generate_attack_schedule
from rs_switcher_common.gp_models import load_gp_switcher, GPSwitcher
from rs_switcher_common.attacks import opt_attack

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

COLOR_PPO  = '#ddeeff'
COLOR_ATLA = '#fde8e6'
COLOR_ATK  = '#f5a623'


def run_traced_episode(perf, backup, controller, attack_start, burst_k, attack_eps,
                       attack_norm='l2', horizon=1000, seed=0):
    rng = np.random.RandomState(seed)
    atk_schedule = np.zeros(horizon, dtype=bool)
    atk_schedule[attack_start: attack_start + burst_k] = True

    obs_ppo = perf.start_episode()
    if hasattr(controller, 'reset_episode'):
        controller.reset_episode()

    R_vals, allow_perfs, step_rewards, phases = [], [], [], []
    prev_reward = 0.0
    done = False
    t = 0

    while not done and t < horizon:
        raw = raw_obs_from_sim(perf.custom_env, perf.config)
        obs_atla = backup.normalize(raw)

        obs_ctrl = obs_ppo
        if atk_schedule[t] and perf.attack_model is not None:
            obs_ctrl = opt_attack(perf.attack_model, obs_ppo,
                                  eps=attack_eps, norm=attack_norm)

        action, info = controller.select(obs_ctrl, obs_atla)
        obs_ppo, _, done, _ = perf.step(action)

        cur_reward = perf.custom_env.total_true_reward
        step_rewards.append(cur_reward - prev_reward)
        prev_reward = cur_reward

        R_vals.append(info.get('R_rs', float('nan')))
        allow_perfs.append(info.get('allow_perf', 1.0))
        phases.append(info.get('phase', 'ppo'))
        t += 1

    return {
        'steps': list(range(t)),
        'R': R_vals,
        'allow_perf': allow_perfs,
        'rewards': step_rewards,
        'attack_start': attack_start,
        'burst_k': burst_k,
        'n_steps': t,
    }


def plot_trace(trace, delta, out_path):
    steps      = np.array(trace['steps'])
    R          = np.array(trace['R'])
    allow_perf = np.array(trace['allow_perf'])
    cum_reward = np.cumsum(trace['rewards'])
    atk_s      = trace['attack_start']
    burst_k    = trace['burst_k']
    T          = trace['n_steps']

    in_ppo = allow_perf >= 0.5   # True = PPO active, False = ATLA active

    fig, axes = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True,
                             gridspec_kw={'height_ratios': [0.8, 2, 2]})

    # ── Panel 1: Policy active stripe ─────────────────────────────────
    ax = axes[0]
    for t in range(T):
        color = COLOR_PPO if in_ppo[t] else COLOR_ATLA
        ax.axvspan(t, t + 1, facecolor=color, linewidth=0)
    ax.axvspan(atk_s, atk_s + burst_k, facecolor=COLOR_ATK, alpha=0.3, zorder=2)
    ax.set_yticks([])
    ax.set_ylabel('Policy', fontsize=9)
    ax.set_xlim(0, T)

    ppo_p  = mpatches.Patch(color=COLOR_PPO,  label='PPO active')
    atla_p = mpatches.Patch(color=COLOR_ATLA, label='ATLA active')
    atk_p  = mpatches.Patch(color=COLOR_ATK,  alpha=0.5, label='Attack window')
    ax.legend(handles=[ppo_p, atla_p, atk_p], loc='upper right',
              fontsize=8, frameon=True, ncol=3, handlelength=1.2)

    # ── Panel 2: Certified radius R ───────────────────────────────────
    ax = axes[1]
    for t in range(T):
        color = COLOR_PPO if in_ppo[t] else COLOR_ATLA
        ax.axvspan(t, t + 1, facecolor=color, alpha=0.5, linewidth=0, zorder=0)
    ax.axvspan(atk_s, atk_s + burst_k, facecolor=COLOR_ATK, alpha=0.15, zorder=1)

    # Replace nan with 0 for plotting
    R_plot = np.where(np.isnan(R), 0, R)
    ax.plot(steps, R_plot, color='#333', linewidth=0.7, alpha=0.8, zorder=3)
    ax.axhline(delta, color='#e05c42', linewidth=1.3, linestyle='--',
               label=f'δ = {delta}', zorder=4)
    ax.set_ylabel('Certified radius R')
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    # ── Panel 3: Cumulative reward ─────────────────────────────────────
    ax = axes[2]
    for t in range(T):
        color = COLOR_PPO if in_ppo[t] else COLOR_ATLA
        ax.axvspan(t, t + 1, facecolor=color, alpha=0.5, linewidth=0, zorder=0)
    ax.axvspan(atk_s, atk_s + burst_k, facecolor=COLOR_ATK, alpha=0.15, zorder=1)
    ax.plot(steps, cum_reward, color='#3aaa6e', linewidth=1.5, zorder=3, label='Switcher')
    ax.set_ylabel('Cumulative Reward')
    ax.set_xlabel('Timestep')
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    axes[0].set_title(
        f'HalfCheetah: Continuous GP Switcher — single episode trace\n'
        f'(attack at t={atk_s}–{atk_s+burst_k}, L2 eps={args.attack_eps}, '
        f'δ={delta}, K_enter={args.K_enter}, K_exit={args.K_exit})',
        pad=5,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f'Saved {out_path.replace(".pdf", ".png")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--attack-path',   default='HalfCheetah/HalfCheetah_Attack_PPO.model')
    p.add_argument('--backup-path',   default='HalfCheetah/HalfCheetah_ATLA.model')
    p.add_argument('--gp-path',       default='models/halfcheetah_switcher_gp_s02.pt')
    p.add_argument('--dataset',       default='data/halfcheetah_critical_dataset.npz')
    p.add_argument('--sigma',         type=float, default=0.2)
    p.add_argument('--delta',         type=float, default=0.2)
    p.add_argument('--K-enter',       type=int,   default=5)
    p.add_argument('--K-exit',        type=int,   default=5)
    p.add_argument('--forgive-decay', type=float, default=1.0)
    p.add_argument('--attack-eps',    type=float, default=0.5)
    p.add_argument('--attack-start',  type=int,   default=300)
    p.add_argument('--burst-k',       type=int,   default=100)
    p.add_argument('--seed',          type=int,   default=0)
    args = p.parse_args()

    config = ENV_REGISTRY['halfcheetah']
    device = torch.device('cpu')

    perf   = MuJoCoPerfPolicy.load(config, args.attack_path, attack_path=args.attack_path)
    backup = MuJoCoBackupPolicy.load(config, args.backup_path)

    data = np.load(args.dataset)
    ckpt = torch.load(args.gp_path, map_location='cpu')
    gp   = GPSwitcher(load_gp_switcher(ckpt), data['state_mean'], data['state_std'],
                      sigma=args.sigma)

    controller = ContinuousSwitcherController(
        perf=perf, backup=backup, rs=gp,
        delta_budget_l2=args.delta,
        K_enter=args.K_enter, K_exit=args.K_exit,
        forgive_decay=args.forgive_decay,
    )

    trace = run_traced_episode(
        perf, backup, controller,
        attack_start=args.attack_start,
        burst_k=args.burst_k,
        attack_eps=args.attack_eps,
        seed=args.seed,
    )

    plot_trace(trace, delta=args.delta,
               out_path='poster_figures/episode_trace.pdf')
