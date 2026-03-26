# RS Certified Switcher

A **runtime-certified binary switcher** for adversarial robustness in RL.

At each step, chooses between a **high-performance PPO policy** and a **safe backup policy** (LQR for CartPole, ATLA for MuJoCo envs). A binary **SwitcherMLP** trained with Randomized Smoothing (RS) detects adversarial observations. For MuJoCo environments, an **AnyTimeSwitcherController** manages transitions via a 4-phase state machine.

## Supported environments

- **CartPole-v1** — PGD L2 attack, quantized LQR backup, Phase-1-only controller
- **Hopper** — Zhang et al. optimal adversary, ATLA backup, 4-phase controller
- **HalfCheetah** — Zhang et al. optimal adversary, ATLA backup, 4-phase controller
- **Walker2D** — Zhang et al. optimal adversary, ATLA backup, 4-phase controller (in progress)

## Project structure

```text
rs_switcher_common/          # Shared infrastructure for MuJoCo envs
  env_config.py              # EnvConfig dataclass + per-env configs
  models.py                  # SwitcherMLP binary classifier
  rs.py                      # VanillaRSSwitcher (GPU-accelerated MC RS)
  controllers.py             # MuJoCoPerfPolicy, MuJoCoBackupPolicy
  evaluation.py              # AnyTimeSwitcherController + evaluate_controller()
  attacks.py                 # opt_attack() (Zhang et al.)
  labeling.py                # CriticalBurstLabeler + collect_state_stats()
  training.py                # train_switcher() (BCE + noise augmentation)
  utils.py, compat.py        # Utilities, gym 0.26 compatibility

cartpole_rs_switcher/        # CartPole-specific (PGD attack, LQR backup)
  controllers.py             # PerfPolicy (SB3 PPO), QuantizedLQRBackup
  evaluation.py              # CertifiedSwitcherController
  attacks.py                 # pgd_l2_attack()
  labeling.py, config.py     # CartPole labeling + config dataclasses
  models.py, rs.py, ...      # Re-exports from rs_switcher_common

scripts/
  build_labels_mujoco.py     # Build dataset (--env hopper|halfcheetah|walker2d)
  train_switcher_mujoco.py   # Train switcher (--env ...)
  evaluate_burst_attack_mujoco.py  # Evaluate controllers (--env ...)
  build_labels.py            # CartPole dataset builder
  train_switcher.py          # CartPole switcher trainer
  evaluate_burst_attack.py   # CartPole evaluation
  train_perf.py              # Train CartPole PPO
  plot_hopper_results.py     # Hopper result figures
  plot_halfcheetah_results.py # HalfCheetah result figures
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start (Hopper example)

```bash
# 1. Build adversarial-detection dataset
python3.8 scripts/build_labels_mujoco.py --env hopper \
    --perf-path Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO.model

# 2. Train binary switcher
python3.8 scripts/train_switcher_mujoco.py --env hopper \
    --dataset data/hopper_critical_dataset.npz \
    --sigma 0.1

# 3. Evaluate under attack
python3.8 scripts/evaluate_burst_attack_mujoco.py --env hopper \
    --perf-path Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --backup-path Hopper/Hopper_ATLA.model \
    --switcher-path models/hopper_switcher.pt \
    --dataset data/hopper_critical_dataset.npz \
    --sigma 0.1 --n-samples 10000 --delta-budget-l2 0.075 \
    --device cuda
```

See `CLAUDE.md` for full documentation including benchmark results and parameter rationale.
