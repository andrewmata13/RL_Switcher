"""
Figure: "The 8 ms Wall" — certified radius vs per-step latency.

Compares GP (single exact point) vs RS at multiple sample counts.
Shade x > 8 ms as "misses real-time deadline".

Output: poster_figures/latency_wall.pdf / .png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('poster_figures/bak_matplotlib.mlpstyle')
plt.rcParams.update({'figure.dpi': 150})


def main():
    data = json.load(open('results/halfcheetah_same_model_h512_s02.json'))
    gp = data['gp']
    rs = data['rs']

    rs_ns = sorted(int(k) for k in rs.keys())
    rs_r = [rs[str(n)]['avg_radius'] for n in rs_ns]
    rs_t = [rs[str(n)]['avg_time_ms'] for n in rs_ns]
    gp_r = gp['avg_radius']
    gp_t = gp['avg_time_ms']

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    DEADLINE_MS = 8.0
    ax.axvspan(DEADLINE_MS, 1e4, facecolor='#fde5e0', alpha=0.85, zorder=0,
               label='Misses 8 ms deadline')
    ax.axvline(DEADLINE_MS, color='#c0392b', linewidth=1.3, linestyle='--',
               zorder=1)
    ax.text(DEADLINE_MS * 1.08, 0.205, '8 ms\ndeadline', color='#c0392b',
            fontsize=10, ha='left', va='bottom', fontweight='bold', zorder=6)

    # GP dotted "matched" radius line
    ax.axhline(gp_r, color='#888', linewidth=0.8, linestyle=':', zorder=1)

    # RS trajectory
    ax.plot(rs_t, rs_r, color='#2c7fb8', linewidth=1.8, marker='o',
            markersize=9, markerfacecolor='#2c7fb8', markeredgecolor='white',
            markeredgewidth=1.2, label='RS (Monte Carlo)', zorder=4)

    for n, t, r in zip(rs_ns, rs_t, rs_r):
        label = f'{n//1000}k' if n >= 1000 else f'{n}'
        # Offset labels to avoid overlap
        dx, dy = 1.12, 0.006
        if n == 1000:
            dx, dy = 1.18, -0.012
        elif n == 100000:
            dx, dy = 0.82, 0.008
        ax.annotate(f'n={label}', (t, r), xytext=(t * dx, r + dy),
                    fontsize=9, color='#2c7fb8',
                    ha='left' if dx > 1 else 'right')

    # GP point
    ax.scatter([gp_t], [gp_r], s=220, marker='*', color='#e07b00',
               edgecolor='black', linewidth=1.0, zorder=6,
               label='GP (exact, ours)')
    ax.annotate('GP', (gp_t, gp_r), xytext=(gp_t * 0.72, gp_r + 0.012),
                fontsize=11, color='#e07b00', fontweight='bold', ha='right')

    ax.set_xscale('log')
    ax.set_xlim(0.2, 250)
    ax.set_ylim(0.22, 0.33)
    ax.set_xlabel('Per-step certification latency (ms, log scale)')
    ax.set_ylabel('Certified radius  R  (larger = tighter)')
    ax.set_title('Certified radius vs. real-time cost (HalfCheetah, σ=0.2)')
    ax.grid(True, which='both', linestyle='--', alpha=0.3, zorder=0)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    os.makedirs('poster_figures', exist_ok=True)
    for ext in ('pdf', 'png'):
        out = f'poster_figures/latency_wall.{ext}'
        plt.savefig(out, bbox_inches='tight')
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
