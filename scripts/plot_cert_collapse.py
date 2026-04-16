"""
Figure: "Certificate Collapse = Detection" — HalfCheetah multi-burst trace.

Two panels stacked:
  (1) Certified radius R(t) with δ line; attack windows shaded red.
      Region fills red below δ when cert fails.
  (2) Policy-active stripe (PPO blue / ATLA orange) with cumulative reward.

Uses multi-burst (3×100 / 100) to show repeated detection & recovery.

Output: poster_figures/cert_collapse.pdf / .png
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
from rs_switcher_common.evaluation import ContinuousSwitcherController
from rs_switcher_common.gp_models import load_gp_switcher, GPSwitcher
from rs_switcher_common.attacks import opt_attack

plt.style.use('poster_figures/bak_matplotlib.mlpstyle')
plt.rcParams.update({'figure.dpi': 150})

COLOR_PPO  = '#4a90d9'
COLOR_ATLA = '#f07b40'
COLOR_ATK  = '#d64545'
COLOR_DELTA = '#333'
COLOR_R    = '#2c3e50'
COLOR_COLLAPSE = '#d64545'


def run_traced(perf, backup, controller, attack_schedule, attack_eps,
               attack_norm='l2', horizon=1000):
    obs_ppo = perf.start_episode()
    if hasattr(controller, 'reset_episode'):
        controller.reset_episode()

    R_vals, p_crit_vals, allow_perfs, step_rewards = [], [], [], []
    prev_reward = 0.0
    done = False
    t = 0
    while not done and t < horizon:
        raw = raw_obs_from_sim(perf.custom_env, perf.config)
        obs_atla = backup.normalize(raw)

        obs_ctrl = obs_ppo
        if attack_schedule[t] and perf.attack_model is not None:
            obs_ctrl = opt_attack(perf.attack_model, obs_ppo,
                                  eps=attack_eps, norm=attack_norm)

        action, info = controller.select(obs_ctrl, obs_atla)
        obs_ppo, _, done, _ = perf.step(action)
        cur = perf.custom_env.total_true_reward
        step_rewards.append(cur - prev_reward)
        prev_reward = cur

        R_vals.append(info.get('R_rs', float('nan')))
        p_crit_vals.append(info.get('p_critical', 0.0))
        allow_perfs.append(info.get('allow_perf', 1.0))
        t += 1

    return dict(R=np.array(R_vals), p_crit=np.array(p_crit_vals),
                allow=np.array(allow_perfs),
                rewards=np.array(step_rewards), T=t)


def plot(trace, schedule, delta, out_path, smooth_w=15, x_range=None):
    T = trace['T']
    steps = np.arange(T)
    R_raw = np.nan_to_num(trace['R'], nan=0.0)
    p_crit = np.nan_to_num(trace['p_crit'], nan=0.0)
    # Signed R: +R when pred==0 (safe direction), −R when pred==1 (attacked)
    sign = np.where(p_crit > 0.5, -1.0, 1.0)
    R_signed_raw = sign * R_raw
    kernel = np.ones(smooth_w) / smooth_w
    R_signed = np.convolve(R_signed_raw, kernel, mode='same')
    in_ppo = trace['allow'] >= 0.5
    cum = np.cumsum(trace['rewards'])

    # Identify attack windows as (start, end) tuples for shading
    atk = schedule[:T].astype(int)
    edges = np.diff(np.concatenate([[0], atk, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    attack_windows = list(zip(starts, ends))

    fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 3.6))

    # ── Panel 1: Signed certified radius ─────────────────────────────
    # Attack windows background
    for (s, e) in attack_windows:
        ax1.axvspan(s, e, facecolor=COLOR_ATK, alpha=0.18, zorder=1,
                    edgecolor='none')

    # Raw signed R as faint dots
    ax1.plot(steps, R_signed_raw, color=COLOR_R, linewidth=0, marker='.',
             markersize=2, alpha=0.22, zorder=2)

    # Smoothed signed R
    ax1.plot(steps, R_signed, color=COLOR_R, linewidth=2.2, zorder=4,
             label=f'Signed R (smoothed, w={smooth_w})')

    # Green fill where certified safe (R >= δ)
    ax1.fill_between(steps, delta, R_signed, where=(R_signed >= delta),
                     color='#2ea05c', alpha=0.25, linewidth=0, zorder=3,
                     label='Certified safe (R ≥ δ)')
    # Red fill where attack-detected (R < 0 means pred=attacked)
    ax1.fill_between(steps, 0, R_signed, where=(R_signed < 0),
                     color=COLOR_COLLAPSE, alpha=0.50, linewidth=0, zorder=3,
                     label='Attack detected (pred = 1)')

    ax1.axhline(delta, color='#2ea05c', linewidth=1.5, linestyle='--',
                zorder=5, alpha=0.9, label=f'δ = {delta} (safe threshold)')
    ax1.axhline(0, color='#555', linewidth=0.8, linestyle='-', zorder=5)

    ax1.set_ylabel('Certified Radius')
    if x_range is not None:
        ax1.set_xlim(*x_range)
    else:
        ax1.set_xlim(0, T)
    lim = max(0.6, float(np.max(np.abs(R_signed))) * 1.1)
    ax1.set_ylim(-lim, lim)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, zorder=0)


    ax1.set_xlabel('Step')
    #ax1.set_title('')

    plt.tight_layout()
    os.makedirs('poster_figures', exist_ok=True)
    for ext in ('pdf', 'png'):
        out = f'poster_figures/cert_collapse.{ext}'
        plt.savefig(out, bbox_inches='tight')
        print(f'Saved {out}')


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
    p.add_argument('--n-bursts',      type=int,   default=3)
    p.add_argument('--burst-k',       type=int,   default=100)
    p.add_argument('--cooldown-k',    type=int,   default=100)
    p.add_argument('--attack-start',  type=int,   default=150)
    p.add_argument('--horizon',       type=int,   default=1000)
    p.add_argument('--seed',          type=int,   default=4)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = ENV_REGISTRY['halfcheetah']
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

    # Build multi-burst schedule
    schedule = np.zeros(args.horizon, dtype=bool)
    t = args.attack_start
    for _ in range(args.n_bursts):
        schedule[t:t + args.burst_k] = True
        t += args.burst_k + args.cooldown_k

    trace = run_traced(perf, backup, controller, schedule,
                       attack_eps=args.attack_eps, horizon=args.horizon)

    print(f"Episode length: {trace['T']}, total reward: {trace['rewards'].sum():.1f}")
    print(f"% in PPO: {trace['allow'].mean()*100:.1f}%")
    plot(trace, schedule, delta=args.delta,
         out_path='poster_figures/cert_collapse.pdf',
         smooth_w=15, x_range=(0, args.horizon))
