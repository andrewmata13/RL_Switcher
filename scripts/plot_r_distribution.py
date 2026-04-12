"""
R distribution: histogram of certified radius R for clean vs attacked observations.
Shows the certificate collapse phenomenon and the delta threshold.
Output: poster_figures/r_distribution.pdf
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

d = json.load(open('results/halfcheetah_R_distribution.json'))
cfg = d['config']
delta = cfg['delta_budget_l2']

clean_R    = np.array(d['clean_R'],    dtype=float)
attacked_R = np.array(d['attacked_R'], dtype=float)
clean_pred    = np.array(d['clean_preds'])
attacked_pred = np.array(d['attacked_preds'])

# Only plot obs where pred==0 (radius is meaningful; pred==1 goes straight to ATLA)
clean_R_pred0    = clean_R[clean_pred == 0]
attacked_R_pred0 = attacked_R[attacked_pred == 0]

fig, ax = plt.subplots(figsize=(5.5, 3.5))

bins = np.linspace(0, 0.7, 60)

ax.hist(clean_R_pred0, bins=bins, density=True, alpha=0.65,
        color='#4a7fc1', label=f'Clean (pred=0, n={len(clean_R_pred0):,})')
ax.hist(attacked_R_pred0, bins=bins, density=True, alpha=0.65,
        color='#e05c42', label=f'Attacked (pred=0, n={len(attacked_R_pred0):,})')

# Delta threshold line
ax.axvline(delta, color='#222', linewidth=1.5, linestyle='--',
           label=f'δ = {delta} (detection threshold)')

# Annotate regions
ymax = ax.get_ylim()[1]
ax.annotate('Certificate\ncollapse\n(detected)', xy=(delta * 0.5, ymax * 0.72),
            ha='center', fontsize=8.5, color='#c0392b',
            arrowprops=None)
ax.annotate('Formally\ncertified', xy=(delta + 0.12, ymax * 0.72),
            ha='center', fontsize=8.5, color='#1a5276')

ax.fill_betweenx([0, ymax * 1.05], 0, delta,
                 color='#e05c42', alpha=0.07, zorder=0)
ax.fill_betweenx([0, ymax * 1.05], delta, bins[-1],
                 color='#4a7fc1', alpha=0.07, zorder=0)

ax.set_ylim(0, ymax * 1.05)
ax.set_xlabel('Certified radius R')
ax.set_ylabel('Density')
ax.set_title(f'HalfCheetah: Certificate Radius Distribution\n(σ={cfg["sigma"]}, δ={delta}, GP switcher, pred=0 steps only)')
ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.35)

# Add pred=1 rate annotation
pred1_clean    = np.mean(clean_pred == 1)
pred1_attacked = np.mean(attacked_pred == 1)
ax.text(0.97, 0.97,
        f'pred=1 rate:\n  clean: {pred1_clean:.1%}\n  attacked: {pred1_attacked:.1%}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
out = 'poster_figures/r_distribution.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
print(f'Saved {out.replace(".pdf", ".png")}')
