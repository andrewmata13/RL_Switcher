"""
Bar chart: Clean vs Attacked return for PPO / ATLA / Continuous GP switcher.
Panels: HalfCheetah (single + multi burst), CartPole (clean + attacked).
Output: poster_figures/bar_chart.pdf
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style ──────────────────────────────────────────────────────────────────
COLORS = {
    'always_perf':    '#e05c42',   # red-orange  — Always PPO
    'always_backup':  '#4a7fc1',   # blue        — Always ATLA / LQR
    'continuous_gp':  '#3aaa6e',   # green       — Ours
    'certified_switcher': '#3aaa6e',
}
LABELS = {
    'always_perf':        'Always PPO',
    'always_backup':      'Always ATLA',
    'continuous_gp':      'Continuous GP (ours)',
    'certified_switcher': 'Certified Switcher (ours)',
}
HATCH_ATTACKED = '//'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_medians(path, ctrl, suffix=None):
    """Load median and IQR from a result JSON.
    Tries '{ctrl}_{suffix}' first (HalfCheetah format),
    then plain '{ctrl}' (CartPole format)."""
    d = json.load(open(path))
    for key in ([f'{ctrl}_{suffix}', ctrl] if suffix else [ctrl]):
        if key in d and 'returns' in d[key]:
            r = d[key]['returns']
            return np.median(r), np.percentile(r, 25), np.percentile(r, 75)
    return None, None, None


def bar_group(ax, groups, controllers, width=0.22, ymin=0, ymax=None):
    """
    groups: list of (label, {ctrl: (med, p25, p75)})
    controllers: ordered list of controller keys to plot
    """
    n_groups = len(groups)
    n_ctrl = len(controllers)
    x = np.arange(n_groups)
    offsets = np.linspace(-(n_ctrl - 1) / 2, (n_ctrl - 1) / 2, n_ctrl) * width

    for i, ctrl in enumerate(controllers):
        meds, yerr_lo, yerr_hi = [], [], []
        for _, vals in groups:
            med, p25, p75 = vals.get(ctrl, (None, None, None))
            meds.append(med if med is not None else 0)
            yerr_lo.append((med - p25) if med is not None else 0)
            yerr_hi.append((p75 - med) if med is not None else 0)

        ax.bar(
            x + offsets[i], meds, width,
            color=COLORS[ctrl], label=LABELS[ctrl],
            yerr=[yerr_lo, yerr_hi],
            error_kw=dict(elinewidth=1, capsize=3, ecolor='#333'),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(ymin, ymax if ymax else None)
    # Broken-axis indicator if y-axis doesn't start at 0
    if ymin > 0:
        d = 0.012  # size of diagonal slash marks
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False)


# ── Data ───────────────────────────────────────────────────────────────────

cheetah_controllers = ['always_perf', 'always_backup', 'continuous_gp']

cheetah_groups = []
for mode_label, fname, suffix in [
    ('Clean',  'results/halfcheetah_paper_multi.json', 'clean'),
    ('Single\nburst', 'results/halfcheetah_paper_single.json', 'attacked'),
    ('Multi\nburst',  'results/halfcheetah_paper_multi.json',  'attacked'),
]:
    vals = {}
    for ctrl in cheetah_controllers:
        med, p25, p75 = load_medians(fname, ctrl, suffix)
        vals[ctrl] = (med, p25, p75)
    cheetah_groups.append((mode_label, vals))

# CartPole — 15k-step PPO, GP switcher, eps=1.0, burst=200 oracle
cartpole_controllers = ['always_perf', 'always_backup', 'certified_switcher']
cartpole_groups = []
for mode_label, fname in [
    ('Clean',              'results/cartpole_15k_gp2_clean.json'),
    ('Attacked\n(burst=200)', 'results/cartpole_15k_gp_bk200.json'),
]:
    vals = {}
    for ctrl in cartpole_controllers:
        med, p25, p75 = load_medians(fname, ctrl)
        vals[ctrl] = (med, p25, p75)
    cartpole_groups.append((mode_label, vals))


# ── Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# HalfCheetah — broken y-axis starting at 2500 to show differences clearly
ax = axes[0]
bar_group(ax, cheetah_groups, cheetah_controllers, width=0.24, ymin=2500, ymax=8500)
ax.set_title('HalfCheetah (L2 eps=0.5, 300 attack steps)')
ax.set_ylabel('Episode Return (median ± IQR)')

# CartPole
ax = axes[1]
bar_group(ax, cartpole_groups, cartpole_controllers, width=0.24, ymax=560)
ax.set_title('CartPole (L2 eps=1.0, burst=200)')
ax.set_ylabel('Episode Return (median ± IQR)')
ax.axhline(500, color='gray', linewidth=0.8, linestyle=':', label='Max return')

# Shared legend
handles = [
    mpatches.Patch(color=COLORS['always_perf'],    label='Always PPO'),
    mpatches.Patch(color=COLORS['always_backup'],  label='Always ATLA / LQR'),
    mpatches.Patch(color=COLORS['continuous_gp'],  label='Certified Switcher (ours)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04),
           frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
out = 'poster_figures/bar_chart.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
print(f'Saved {out.replace(".pdf", ".png")}')
