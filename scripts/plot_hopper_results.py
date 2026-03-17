"""
Generate figures and summary table for Hopper burst-attack evaluation.

Reads results/hopper_seed{0,42,123}.json and produces:
  figures/hopper_return_comparison.pdf   — grouped bar chart (clean vs attacked)
  figures/hopper_fall_rate.pdf           — fall rate bar chart
  figures/hopper_return_distribution.pdf — box plots of per-episode returns

Usage:
    python3.8 scripts/plot_hopper_results.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEEDS    = [0, 42, 123]
EPISODES = 30
RESULTS_DIR = "results"
FIG_DIR     = "figures"

CONTROLLERS = ["always_perf", "always_backup", "anytime_switcher"]
LABELS      = {"always_perf": "Always PPO", "always_backup": "Always ATLA",
               "anytime_switcher": "Anytime Switcher"}
COLORS      = {"always_perf": "#4C72B0", "always_backup": "#DD8452",
               "anytime_switcher": "#55A868"}

os.makedirs(FIG_DIR, exist_ok=True)


def load_results():
    """Aggregate per-seed JSON files."""
    agg = {}   # key -> list of per-seed dicts
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"hopper_seed{seed}.json")
        with open(path) as f:
            data = json.load(f)
        for key, m in data.items():
            agg.setdefault(key, []).append(m)
    return agg


def print_table(agg):
    header = (f"{'Controller':<22}  {'Condition':<8}  "
              f"{'Return':>14}  {'Falls':>10}  {'allow_perf':>10}  {'R_exec':>8}")
    print(header)
    print("-" * len(header))
    for ctrl in CONTROLLERS:
        for cond in ["clean", "attacked"]:
            key = f"{ctrl}_{cond}"
            all_returns  = [r for m in agg[key] for r in m["returns"]]
            ret_mean     = float(np.mean(all_returns))
            ret_std      = float(np.std(all_returns))
            fall_mean    = float(np.mean([m["fall_rate"] for m in agg[key]]))
            allow_mean   = float(np.mean([m["mean_allow_perf"] for m in agg[key]]))
            R_vals       = [m["mean_R_exec"] for m in agg[key] if not np.isnan(m["mean_R_exec"])]
            R_mean       = float(np.nanmean(R_vals)) if R_vals else float("nan")
            n_fell       = int(round(fall_mean * EPISODES * len(SEEDS)))
            total        = EPISODES * len(SEEDS)
            print(f"{LABELS[ctrl]:<22}  {cond:<8}  "
                  f"{ret_mean:>7.0f}±{ret_std:<5.0f}  "
                  f"{n_fell:>4}/{total:<5}  "
                  f"{allow_mean:>10.3f}  "
                  f"{R_mean:>8.4f}")
    print()


def plot_return_comparison(agg):
    fig, ax = plt.subplots(figsize=(8, 5))

    n = len(CONTROLLERS)
    x = np.arange(n)
    width = 0.32
    offsets = [-width/2, width/2]

    for i, (cond, label, hatch) in enumerate([("clean", "Clean", ""), ("attacked", "Attacked", "//")]):
        means, errs = [], []
        for ctrl in CONTROLLERS:
            key = f"{ctrl}_{cond}"
            all_returns = [r for m in agg[key] for r in m["returns"]]
            means.append(np.mean(all_returns))
            # 95% CI: std / sqrt(N) * 1.96
            errs.append(1.96 * np.std(all_returns) / np.sqrt(len(all_returns)))

        bars = ax.bar(x + offsets[i], means, width, yerr=errs,
                      capsize=4, hatch=hatch,
                      color=[COLORS[c] for c in CONTROLLERS],
                      alpha=0.85 if cond == "clean" else 0.55,
                      edgecolor="black", linewidth=0.7,
                      label=label, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONTROLLERS], fontsize=11)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("Hopper: Episode Return under Burst Attack\n"
                 r"($\epsilon_{L\infty}=0.075$, burst_k=50, t_max=300, 30 eps × 3 seeds)",
                 fontsize=11)
    ax.set_ylim(0, 4200)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Legend: clean/attacked pattern + controller colors
    clean_patch    = mpatches.Patch(facecolor="grey", alpha=0.85, label="Clean")
    attacked_patch = mpatches.Patch(facecolor="grey", alpha=0.55, hatch="//",
                                    edgecolor="black", label="Attacked")
    color_patches  = [mpatches.Patch(facecolor=COLORS[c], label=LABELS[c])
                      for c in CONTROLLERS]
    ax.legend(handles=[clean_patch, attacked_patch] + color_patches,
              fontsize=9, loc="upper right", ncol=2)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "hopper_return_comparison.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_fall_rate(agg):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n = len(CONTROLLERS)
    x = np.arange(n)
    width = 0.32
    offsets = [-width/2, width/2]

    for i, (cond, hatch) in enumerate([("clean", ""), ("attacked", "//")]):
        fall_rates, errs = [], []
        for ctrl in CONTROLLERS:
            key = f"{ctrl}_{cond}"
            per_seed = [m["fall_rate"] for m in agg[key]]
            fall_rates.append(np.mean(per_seed))
            errs.append(1.96 * np.std(per_seed) / np.sqrt(len(per_seed)) if len(per_seed) > 1 else 0)

        ax.bar(x + offsets[i], [r * 100 for r in fall_rates], width,
               yerr=[e * 100 for e in errs], capsize=4, hatch=hatch,
               color=[COLORS[c] for c in CONTROLLERS],
               alpha=0.85 if cond == "clean" else 0.55,
               edgecolor="black", linewidth=0.7,
               label=cond.capitalize(), error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONTROLLERS], fontsize=11)
    ax.set_ylabel("Fall Rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("Hopper: Episode Fall Rate under Burst Attack", fontsize=11)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    clean_patch    = mpatches.Patch(facecolor="grey", alpha=0.85, label="Clean")
    attacked_patch = mpatches.Patch(facecolor="grey", alpha=0.55, hatch="//",
                                    edgecolor="black", label="Attacked")
    color_patches  = [mpatches.Patch(facecolor=COLORS[c], label=LABELS[c])
                      for c in CONTROLLERS]
    ax.legend(handles=[clean_patch, attacked_patch] + color_patches,
              fontsize=9, loc="upper right", ncol=2)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "hopper_fall_rate.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_return_distribution(agg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, cond in zip(axes, ["clean", "attacked"]):
        data  = []
        ticks = []
        for ctrl in CONTROLLERS:
            key = f"{ctrl}_{cond}"
            all_returns = [r for m in agg[key] for r in m["returns"]]
            data.append(all_returns)
            ticks.append(LABELS[ctrl])

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, ctrl in zip(bp["boxes"], CONTROLLERS):
            patch.set_facecolor(COLORS[ctrl])
            patch.set_alpha(0.8)

        ax.set_xticklabels(ticks, fontsize=10, rotation=10)
        ax.set_title(f"{'Clean' if cond == 'clean' else 'Under Attack'}", fontsize=12)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        if cond == "clean":
            ax.set_ylabel("Episode Return", fontsize=12)

    fig.suptitle("Hopper Return Distribution (90 episodes per controller, 3 seeds)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "hopper_return_distribution.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    agg = load_results()
    print("=== Hopper Evaluation Summary (3 seeds × 30 episodes) ===\n")
    print_table(agg)
    plot_return_comparison(agg)
    plot_fall_rate(agg)
    plot_return_distribution(agg)


if __name__ == "__main__":
    main()
