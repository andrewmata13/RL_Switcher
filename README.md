# RS Certified Switcher

A **runtime-certified binary switcher** for adversarial robustness in RL. Submitted to ACNS 2026.

At each step, chooses between a **high-performance PPO policy** and a **safe backup** (LQR for CartPole, ATLA for MuJoCo). A binary switcher trained with Randomized Smoothing (RS) detects adversarial observations and triggers backup via a hysteresis controller, with no permanent commitment — supporting repeated and multi-burst attacks.

## Environments

| Env | Backup | Attack | Status |
|-----|--------|--------|--------|
| CartPole-v1 | Quantized LQR | PGD L2, eps=1.0 | Complete (paper) |
| HalfCheetah | ATLA | Zhang et al. L2, eps=0.5 | Complete (paper) |
| Hopper | ATLA / PPOAsBackup | Zhang et al. L2 | Secondary stories |
| Walker2D | ATLA | Zhang et al. L2 | Incomplete |

## Project structure

```
rs_switcher_common/          # Shared MuJoCo infrastructure
  env_config.py              # EnvConfig + HOPPER/HALFCHEETAH/WALKER2D registry
  models.py                  # SwitcherMLP, SwitcherDeepMLP, SwitcherRobustMLP
  gp_models.py               # SwitcherQuadMLP (x²+x), GPSwitcher (Gil-Pelaez cert)
  rs.py                      # VanillaRSSwitcher (MC Clopper-Pearson)
  controllers.py             # MuJoCoPerfPolicy, MuJoCoBackupPolicy, raw_obs_from_sim()
  evaluation.py              # ContinuousSwitcherController, AdaptiveSwitcherController
  attacks.py                 # opt_attack() — Zhang et al. L-inf and L2
  labeling.py                # CriticalBurstLabeler, collect_state_stats()
  training.py                # train_switcher() — BCE + noise augmentation
  clean_policies.py          # CleanPerfPolicy, PPOAsBackup
  compat.py                  # gym 0.26 compatibility

cartpole_rs_switcher/        # CartPole-specific
  controllers.py             # PerfPolicy (SB3 PPO), QuantizedLQRBackup
  evaluation.py              # CertifiedSwitcherController
  attacks.py                 # pgd_l2_attack()

scripts/
  evaluate_continuous_controller.py  # Main MuJoCo eval (all attack modes)
  evaluate_burst_attack.py           # CartPole eval
  build_labels_mujoco.py             # MuJoCo dataset builder
  build_labels_cartpole.py           # CartPole dataset builder (MuJoCo-style)
  train_switcher_gp.py               # Train SwitcherQuadMLP GP switcher
  train_switcher.py                  # Train MC RS switcher
  train_perf.py                      # Train CartPole PPO
  plot_cert_collapse.py              # poster_figures/cert_collapse.pdf (signed R trace)
  plot_latency_wall.py               # poster_figures/latency_wall.pdf (GP vs RS timing)
  plot_bar_chart.py                  # poster_figures/bar_chart.pdf
  plot_r_distribution.py             # poster_figures/r_distribution.pdf
  plot_episode_trace.py              # poster_figures/episode_trace.pdf
  evaluate_cartpole_multi_arb.py     # CartPole multi-burst / arbitrary eval
  compare_gp_vs_rs_same_model.py     # GP vs RS timing benchmark

data/                        # Datasets (.npz): X, y, state_mean, state_std
models/                      # Switcher checkpoints (.pt)
results/                     # Evaluation JSONs
poster_figures/              # Paper figures (PDF + PNG)
HalfCheetah/, Hopper/, Walker2D/  # Pre-trained policy checkpoints (.model/.pt)
policy_gradients/            # Required for unpickling .model files — do not delete
```

## Key results

**HalfCheetah** (L2 eps=0.5, 300 total attack steps, 30 episodes):

| Controller | Clean | Single burst | Multi-burst (3×100) | Arbitrary (E=300) |
|---|---|---|---|---|
| Always PPO | 7245 ± 99 | 4082 ± 1290 | 3168 ± 1661 | 5929 ± 1210 |
| Always ATLA | 5652 ± 46 | 5641 ± 50 | 5649 ± 52 | 5632 ± 44 |
| **Continuous GP (ours)** | **7168 ± 125** | **6687 ± 135** | **6598 ± 1196** | **5874 ± 1934** |

**CartPole** (L2 eps=1.0, 50 episodes):

| Controller | Clean | Single (burst=200) | Multi (2×100) | Arbitrary (E=200) |
|---|---|---|---|---|
| Always PPO | 500 ± 0 | 107 ± 42 | 450 ± 98 | 500 ± 0 |
| Always LQR | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 |
| **GP Switcher (ours)** | **500 ± 0** | **500 ± 0** | **500 ± 0** | **500 ± 0** |

## Quick start (HalfCheetah)

```bash
# Evaluate continuous GP switcher under multi-burst attack
python3.8 scripts/evaluate_continuous_controller.py --env halfcheetah \
    --perf-path HalfCheetah/HalfCheetah_PPO.model \
    --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --backup-path HalfCheetah/HalfCheetah_ATLA.model \
    --gp-switcher-path models/halfcheetah_switcher_gp_s02.pt \
    --dataset data/halfcheetah_critical_dataset.npz \
    --sigma 0.2 --delta-budget-l2 0.2 --episodes 30 --seed 0 \
    --attack-mode multi --n-bursts 3 --burst-k 100 --cooldown-k 100 \
    --K-enter 5 --K-exit 5 --forgive-decay 1.0 \
    --attack-norm l2 --attack-eps 0.5

# Regenerate paper figures
python3.8 scripts/plot_cert_collapse.py --seed 4
python3.8 scripts/plot_latency_wall.py
python3.8 scripts/plot_bar_chart.py
python3.8 scripts/plot_r_distribution.py
```

See `CLAUDE.md` for full documentation, parameter rationale, and per-environment details.
