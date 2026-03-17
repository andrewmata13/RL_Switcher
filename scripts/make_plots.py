import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cartpole_ags_rs_switcher.attacks import pgd_l2_attack
from cartpole_ags_rs_switcher.controllers import PerfPolicy, QuantizedLQRBackup
from cartpole_ags_rs_switcher.evaluation import (
    AlwaysPerfController, AlwaysBackupController,
    UncertifiedSwitcherController, CertifiedSwitcherController,
)
from cartpole_ags_rs_switcher.models import SwitcherMLP
from cartpole_ags_rs_switcher.rs import VanillaRSSwitcher


CONTROLLER_LABELS = ["always_perf", "always_backup", "uncertified_switcher", "certified_switcher"]
CONTROLLER_COLORS = ["#e74c3c", "#2ecc71", "#f39c12", "#3498db"]


# ── helpers ───────────────────────────────────────────────────────────────────

def rollout_episode(env_id, controller, perf_policy, seed, epsilon_l2,
                    state_mean, state_std, burst_k, burst_start=50,
                    attacked=True, horizon=500):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    total, done, t = 0.0, False, 0
    log = []
    while not done and t < horizon:
        obs_f = np.array(obs, dtype=np.float32)
        in_burst = attacked and (burst_start <= t < burst_start + burst_k)
        if in_burst:
            obs_f = pgd_l2_attack(perf_policy, obs_f, epsilon_l2,
                                   state_mean, state_std, n_steps=10)
        action, info = controller.select(obs_f)
        obs, r, term, trunc, _ = env.step(action)
        total += r
        done = term or trunc
        log.append({
            "t": t, "obs": np.array(obs, dtype=np.float32),
            "in_burst": in_burst, "allow_perf": info["allow_perf"],
            "p_critical": info.get("p_critical", np.nan),
            "R_rs": info.get("R_rs", np.nan),
        })
        t += 1
    env.close()
    return total, log


def run_episodes(env_id, controller, perf_policy, n_ep, seed,
                 epsilon_l2, state_mean, state_std, burst_k,
                 attacked, burst_start=50):
    returns = []
    for ep in range(n_ep):
        ret, _ = rollout_episode(env_id, controller, perf_policy,
                                  seed + ep, epsilon_l2, state_mean, state_std,
                                  burst_k, burst_start=burst_start, attacked=attacked)
        returns.append(ret)
    return np.array(returns)


# ── Plot 1: Return comparison bar chart ───────────────────────────────────────

def plot_return_comparison(controllers, perf, env_id, epsilon_l2,
                           state_mean, state_std, burst_k,
                           n_ep, seed, out_path):
    print("Collecting returns for comparison plot...")
    results = {}
    for name, ctrl in controllers.items():
        r_clean  = run_episodes(env_id, ctrl, perf, n_ep, seed,
                                epsilon_l2, state_mean, state_std,
                                burst_k, attacked=False)
        r_attack = run_episodes(env_id, ctrl, perf, n_ep, seed,
                                epsilon_l2, state_mean, state_std,
                                burst_k, attacked=True)
        results[name] = (r_clean, r_attack)
        print(f"  {name}: clean={r_clean.mean():.1f}  attack={r_attack.mean():.1f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONTROLLER_LABELS))
    w = 0.35

    for i, name in enumerate(CONTROLLER_LABELS):
        c_vals, a_vals = results[name]
        col = CONTROLLER_COLORS[i]
        ax.bar(x[i] - w/2, c_vals.mean(), w, color=col, alpha=0.40,
               edgecolor=col, linewidth=1.5)
        ax.bar(x[i] + w/2, a_vals.mean(), w, color=col, alpha=0.90,
               edgecolor=col, linewidth=1.5)
        ax.errorbar(x[i] - w/2, c_vals.mean(), yerr=c_vals.std(),
                    fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.errorbar(x[i] + w/2, a_vals.mean(), yerr=a_vals.std(),
                    fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Always\nPerf (PPO)", "Always\nBackup (LQR)",
                         "Uncertified\nSwitcher", "Certified\nSwitcher"], fontsize=11)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title(
        f"Controller Performance: Clean vs. Burst Attack\n"
        f"(ε={epsilon_l2}, burst_k={burst_k}, burst_start=50, {n_ep} episodes)\n"
        f"Certified switcher improves both clean performance and adversarial robustness\n"
        f"(Weak PPO: ~363 clean due to 10k training steps; LQR backup compensates)",
        fontsize=10,
    )
    ax.set_ylim(0, 570)
    ax.axhline(500, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(x[-1] + 0.55, 503, "max (500)", color="gray", fontsize=9, va="bottom")

    legend_patches = [
        mpatches.Patch(color="gray", alpha=0.40, label="Clean (no attack)"),
        mpatches.Patch(color="gray", alpha=0.90, label="Under burst attack"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 2: PPO clean vs PPO under attack (side-by-side trajectories) ─────────

def plot_ppo_attack_comparison(perf_ctrl, perf_policy, env_id, epsilon_l2,
                               state_mean, state_std, burst_k,
                               seed, burst_start, out_path):
    print("Collecting PPO clean vs attack trajectories...")

    _, log_clean  = rollout_episode(env_id, perf_ctrl, perf_policy, seed,
                                    epsilon_l2, state_mean, state_std,
                                    burst_k, burst_start=burst_start, attacked=False)
    _, log_attack = rollout_episode(env_id, perf_ctrl, perf_policy, seed,
                                    epsilon_l2, state_mean, state_std,
                                    burst_k, burst_start=burst_start, attacked=True)

    def extract(log):
        ts    = np.array([s["t"] for s in log])
        theta = np.degrees(np.array([s["obs"][2] for s in log]))
        x_pos = np.array([s["obs"][0] for s in log])
        burst = np.array([s["in_burst"] for s in log])
        return ts, theta, x_pos, burst

    ts_c, theta_c, x_c, _       = extract(log_clean)
    ts_a, theta_a, x_a, burst_a = extract(log_attack)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    fig.suptitle(
        f"PPO (Weak) — Clean vs. Burst Attack\n"
        f"(ε={epsilon_l2}, burst_k={burst_k}, burst_start={burst_start})",
        fontsize=12,
    )

    burst_span_kw = dict(color="#f39c12", alpha=0.30, label="Burst attack window")

    for ax, ts, theta, label, color, burst in [
        (axes[0], ts_c, theta_c, "Clean (no attack)", "#3498db", None),
        (axes[1], ts_a, theta_a, f"Under attack (ε={epsilon_l2})", "#e74c3c", burst_a),
    ]:
        ax.plot(ts, theta, color=color, linewidth=1.8, label=label)
        ax.axhline(0,   color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline( 12, color="#e74c3c", linestyle=":", linewidth=0.9, alpha=0.6,
                    label="Failure threshold (±12°)")
        ax.axhline(-12, color="#e74c3c", linestyle=":", linewidth=0.9, alpha=0.6)

        if burst is not None:
            first = True
            for t, b in zip(ts, burst):
                if b:
                    kw = burst_span_kw if first else dict(color="#f39c12", alpha=0.30)
                    ax.axvspan(t - 0.5, t + 0.5, **kw)
                    first = False

        ax.set_ylabel("Pole Angle (°)", fontsize=10)
        ax.set_ylim(-20, 20)
        end_step = int(ts[-1])
        ax.set_xlim(0, max(end_step + 5, burst_start + burst_k + 30))
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ret_label = f"Episode length: {end_step + 1} steps"
        ax.text(0.02, 0.05, ret_label, transform=ax.transAxes,
                fontsize=9, color="gray")

    axes[1].set_xlabel("Timestep", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 3: Certified switcher episode timeline ───────────────────────────────

def plot_episode_timeline(certified_ctrl, perf, env_id, epsilon_l2,
                          state_mean, state_std, burst_k,
                          seed, burst_start, delta_budget, out_path):
    print("Collecting certified switcher episode timeline...")
    _, log = rollout_episode(env_id, certified_ctrl, perf, seed,
                              epsilon_l2, state_mean, state_std,
                              burst_k, burst_start=burst_start, attacked=True)

    ts         = np.array([s["t"] for s in log])
    theta      = np.degrees(np.array([s["obs"][2] for s in log]))
    in_burst   = np.array([s["in_burst"] for s in log])
    allow_perf = np.array([s["allow_perf"] for s in log])
    p_crit     = np.array([s["p_critical"] for s in log])
    R_rs       = np.array([s["R_rs"] for s in log])

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Certified Switcher — Episode Timeline Under Burst Attack\n"
        f"(ε={epsilon_l2}, burst_k={burst_k}, burst_start={burst_start}, δ={delta_budget})",
        fontsize=12,
    )

    def shade_burst(ax):
        first = True
        for t, b in zip(ts, in_burst):
            if b:
                kw = dict(color="#f39c12", alpha=0.25, label="Burst attack") if first \
                     else dict(color="#f39c12", alpha=0.25)
                ax.axvspan(t - 0.5, t + 0.5, **kw)
                first = False

    # pole angle
    ax = axes[0]
    ax.plot(ts, theta, color="#2c3e50", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    shade_burst(ax)
    ax.set_ylabel("Pole Angle (°)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # controller selection
    ax = axes[1]
    perf_steps   = ts[allow_perf == 1]
    backup_steps = ts[allow_perf == 0]
    ax.scatter(perf_steps,   np.ones_like(perf_steps),   color="#3498db",
               marker="|", s=200, linewidths=1.5, label="Perf (PPO)")
    ax.scatter(backup_steps, np.ones_like(backup_steps), color="#e74c3c",
               marker="|", s=200, linewidths=1.5, label="Backup (LQR)")
    shade_burst(ax)
    ax.set_yticks([])
    ax.set_ylabel("Controller", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # p_critical
    ax = axes[2]
    ax.plot(ts, p_crit, color="#e74c3c", linewidth=1.5, label="p_critical")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    shade_burst(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("p_critical", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # certified radius R
    ax = axes[3]
    valid = ~np.isnan(R_rs)
    ax.plot(ts[valid], R_rs[valid], color="#3498db", linewidth=1.5, label="Certified R")
    ax.axhline(delta_budget, color="#e74c3c", linestyle="--",
               linewidth=1.2, label=f"δ budget = {delta_budget}")
    shade_burst(ax)
    ax.set_ylabel("Certified R (norm. L2)", fontsize=10)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 4: State-space criticality map ──────────────────────────────────────

def plot_state_space(rs, dataset_path, delta_budget, out_path):
    print("Computing state-space criticality map...")
    data = np.load(dataset_path)
    X, y = data["X"], data["y"]

    R_vals = []
    for obs in X:
        pred, p_lower, R = rs.certify(obs)
        R_vals.append(R if pred == 0 else 0.0)
    R_vals = np.array(R_vals)

    theta     = np.degrees(X[:, 2])
    theta_dot = X[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("State-Space Criticality Map (θ vs θ̇)", fontsize=13)

    # left: ground-truth labels
    ax = axes[0]
    nc, cr = y == 0, y == 1
    ax.scatter(theta[nc], theta_dot[nc], c="#3498db", s=18, alpha=0.6,
               label=f"Non-critical (n={nc.sum()})")
    ax.scatter(theta[cr], theta_dot[cr], c="#e74c3c", s=18, alpha=0.6,
               label=f"Critical (n={cr.sum()})")
    ax.set_xlabel("Pole Angle θ (°)", fontsize=11)
    ax.set_ylabel("Angular Velocity θ̇ (rad/s)", fontsize=11)
    ax.set_title("Ground-Truth Labels", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # right: certified radius for non-critical states, grey for critical
    ax = axes[1]
    ax.scatter(theta[cr], theta_dot[cr], c="#cccccc", s=18, alpha=0.5,
               label="Critical → Backup", zorder=1)
    R_nc = R_vals[nc]
    sc = ax.scatter(theta[nc], theta_dot[nc], c=R_nc,
                    cmap="Blues", vmin=0, vmax=max(R_nc.max(), delta_budget + 0.05),
                    s=25, alpha=0.85, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Certified R (norm. L2)", fontsize=10)
    cbar.ax.axhline(delta_budget, color="#e74c3c", linewidth=1.5, linestyle="--")
    cbar.ax.text(1.08, delta_budget / (R_nc.max() + 0.05),
                 f"δ={delta_budget}", transform=cbar.ax.transAxes,
                 color="#e74c3c", fontsize=9, va="center")

    # annotate the certified safe zone (R >= delta)
    n_certified = int((R_nc >= delta_budget).sum())
    n_uncertified = int((R_nc < delta_budget).sum())
    ax.set_xlabel("Pole Angle θ (°)", fontsize=11)
    ax.set_ylabel("Angular Velocity θ̇ (rad/s)", fontsize=11)
    ax.set_title(
        f"Certified Radius R\n"
        f"R≥δ (use PPO): {n_certified}  |  R<δ (use backup): {n_uncertified + cr.sum()}",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-path",       type=str,   default="models/perf_cartpole_weak")
    parser.add_argument("--switcher-path",   type=str,   default="models/switcher.pt")
    parser.add_argument("--dataset",         type=str,   default="data/critical_dataset.npz")
    parser.add_argument("--env-id",          type=str,   default="CartPole-v1")
    parser.add_argument("--sigma",           type=float, default=0.25)
    parser.add_argument("--n-samples",       type=int,   default=1000)
    parser.add_argument("--epsilon-l2",      type=float, default=0.5)
    parser.add_argument("--delta-budget-l2", type=float, default=0.5)
    parser.add_argument("--burst-k",         type=int,   default=10)
    parser.add_argument("--burst-start",     type=int,   default=50,
                        help="Attack burst start step for timeline/comparison plots")
    parser.add_argument("--episodes",        type=int,   default=30)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--timeline-seed",   type=int,   default=5)
    parser.add_argument("--out-dir",         type=str,   default="figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perf   = PerfPolicy.load(args.perf_path, device=device)
    backup = QuantizedLQRBackup()
    data   = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]

    ckpt     = torch.load(args.switcher_path, map_location="cpu")
    switcher = SwitcherMLP(obs_dim=int(ckpt["obs_dim"]), hidden_dim=int(ckpt["hidden_dim"]))
    switcher.load_state_dict(ckpt["state_dict"])
    switcher.eval()

    rs = VanillaRSSwitcher(switcher, mean, std, sigma=args.sigma,
                           n_samples=args.n_samples, confidence=0.001)

    controllers = {
        "always_perf":          AlwaysPerfController(perf),
        "always_backup":        AlwaysBackupController(backup),
        "uncertified_switcher": UncertifiedSwitcherController(perf, backup, rs),
        "certified_switcher":   CertifiedSwitcherController(perf, backup, rs,
                                    delta_budget_l2=args.delta_budget_l2),
    }

    # Plot 1: return comparison
    plot_return_comparison(
        controllers, perf, args.env_id,
        args.epsilon_l2, mean, std, args.burst_k,
        args.episodes, args.seed,
        os.path.join(args.out_dir, "return_comparison.png"),
    )

    # Plot 2: PPO clean vs under attack
    plot_ppo_attack_comparison(
        controllers["always_perf"], perf, args.env_id,
        args.epsilon_l2, mean, std, args.burst_k,
        args.timeline_seed, args.burst_start,
        os.path.join(args.out_dir, "ppo_attack_vs_clean.png"),
    )

    # Plot 3: certified switcher timeline
    plot_episode_timeline(
        controllers["certified_switcher"], perf, args.env_id,
        args.epsilon_l2, mean, std, args.burst_k,
        args.timeline_seed, args.burst_start, args.delta_budget_l2,
        os.path.join(args.out_dir, "episode_timeline.png"),
    )

    # Plot 4: state-space criticality map
    plot_state_space(
        rs, args.dataset, args.delta_budget_l2,
        os.path.join(args.out_dir, "state_space_criticality.png"),
    )

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
