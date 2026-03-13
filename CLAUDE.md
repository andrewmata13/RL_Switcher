# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Use `python3.8` (has all dependencies pre-installed). Run all scripts from the repo root.

```bash
pip install -r requirements.txt  # if setting up a new environment
```

Dependencies: `gymnasium`, `numpy`, `scipy`, `torch`, `stable-baselines3`.

## Workflow

```bash
# 1. Train performance PPO policy
python3.8 scripts/train_perf.py --output models/perf_cartpole_ppo.zip

# 2. Build critical-state labels dataset
#    epsilon-l2 is in NORMALIZED observation space; burst-k=25 gives ~30-50% critical
python3.8 scripts/build_labels.py \
  --perf-path models/perf_cartpole_ppo \
  --dataset-out data/critical_dataset.npz \
  --epsilon-l2 1.0 \
  --burst-k 25 \
  --reward-drop-threshold 30 \
  --pgd-steps 5 \
  --n-attack-starts 3

# 3. Train binary switcher with RS noise augmentation (sigma in normalized space)
python3.8 scripts/train_switcher.py \
  --dataset data/critical_dataset.npz \
  --output models/switcher.pt \
  --hidden-dim 64 \
  --epochs 500 \
  --sigma 0.25

# 4. Evaluate certified switching (basic)
python3.8 scripts/evaluate_switcher.py \
  --perf-path models/perf_cartpole_ppo \
  --switcher-path models/switcher.pt \
  --dataset data/critical_dataset.npz \
  --sigma 0.25 \
  --n-samples 2000

# 5. Evaluate under burst attacks (compares all controller variants)
#    epsilon-l2 and delta-budget-l2 are both in NORMALIZED observation space
python3.8 scripts/evaluate_burst_attack.py \
  --perf-path models/perf_cartpole_ppo \
  --switcher-path models/switcher.pt \
  --dataset data/critical_dataset.npz \
  --sigma 0.25 \
  --n-samples 2000 \
  --delta-budget-l2 1.0 \
  --epsilon-l2 1.0 \
  --burst-k 60 \
  --attack-mode fixed
```

Pre-built artifacts are in `models/` and `data/`:
- `data/critical_dataset.npz`
- `models/perf_cartpole_ppo.zip`
- `models/switcher.pt`

**Note:** `stable-baselines3` appends `.zip` automatically, so pass PPO paths **without** the `.zip` extension (e.g. `models/perf_cartpole_ppo`, not `models/perf_cartpole_ppo.zip`).

## Architecture

The system is a **runtime-certified binary switcher** for `CartPole-v1`. At each step it chooses between a high-performance PPO policy and a safe LQR backup by checking whether the current observation is "certified non-critical."

### Core decision rule

Use the performance policy iff:
- The RS-smoothed switcher predicts **non-critical** (`cA == 0`)
- The certified radius `R >= delta_budget_l2` (the adversary's L2 budget, both in **normalized obs space**)

### Component pipeline

```
Raw observation (4D)
    │
    ├──► normalize(obs, state_mean, state_std)
    │
    ├──► VanillaRSSwitcher.certify()
    │        Samples n_samples noisy copies in NORMALIZED obs space (sigma_norm),
    │        runs SwitcherMLP on each, Clopper-Pearson lower bound → p_A_lower.
    │        R = sigma * Phi^{-1}(p_A_lower) in normalized L2 space.
    │
    ├──► CertifiedSwitcherController.select()
    │        allow_perf = (pred == 0) and (R >= delta_budget_l2)
    │        Returns action + diagnostic dict.
    │
    ├──► PerfPolicy (PPO via stable-baselines3)
    └──► QuantizedLQRBackup (discrete-time LQR, Riccati solution)
```

### Key files

| File | Role |
|------|------|
| `cartpole_ags_rs_switcher/models.py` | `SwitcherMLP`: 1-hidden-layer binary classifier (normalized obs → logit) |
| `cartpole_ags_rs_switcher/rs.py` | `VanillaRSSwitcher`: MC randomized smoothing in normalized obs space; `certify()` returns `(pred, p_A_lower, R_norm)` |
| `cartpole_ags_rs_switcher/evaluation.py` | `CertifiedSwitcherController` + `AlwaysPerf/Backup/Uncertified` + `evaluate_controller()` |
| `cartpole_ags_rs_switcher/controllers.py` | `PerfPolicy` (PPO wrapper) and `QuantizedLQRBackup` |
| `cartpole_ags_rs_switcher/labeling.py` | `CriticalBurstLabeler`: rolls out PGD burst attacks (normalized space) to label states critical/non-critical |
| `cartpole_ags_rs_switcher/attacks.py` | `pgd_l2_attack()`: PGD L2 attack in **normalized obs space**; takes `state_mean/std`; returns perturbed obs in raw space |
| `cartpole_ags_rs_switcher/training.py` | `train_switcher()`: trains `SwitcherMLP` with BCE + noise augmentation in normalized space |
| `cartpole_ags_rs_switcher/config.py` | Dataclasses: `LabelConfig`, `SwitcherTrainConfig`, `EvalConfig` |
| `cartpole_ags_rs_switcher/ags.py` | `RegionizedAGSSwitcher`: old AGS certifier — kept for reference, no longer used at runtime |
| `scripts/evaluate_burst_attack.py` | Evaluation comparing `AlwaysPerf`, `AlwaysBackup`, `UncertifiedSwitcher`, `CertifiedSwitcher` under attack |

### Observation space note

CartPole state_std ≈ `[0.045, 0.136, 0.0055, 0.202]`. The pole_angle std is ~25× smaller than cart_vel. All attack and RS quantities are in **normalized L2 space** to avoid the raw-space mismatch where sigma=0.1 raw is 18× the pole_angle std.

### Important conventions

- **`p1` = P(critical)**, not P(allow perf). The switcher predicts criticality; permission requires `pred == 0` (non-critical).
- All L2 quantities (`epsilon_l2`, `delta_budget_l2`, RS `sigma`, certified `R`) are in **normalized observation space**.
- `pos_weight = neg / pos` in training: up-weights the critical class regardless of class balance direction.
- The switcher checkpoint format is `{"state_dict": ..., "obs_dim": int, "hidden_dim": int}`.
- The dataset `.npz` format has keys `X`, `y`, `state_mean`, `state_std`.

### Active issue (as of 2026-03-13)

The normalized-space PGD attack (`pgd_l2_attack`) flips ~38% of individual PPO actions but fails to physically destabilize CartPole over a burst (0% critical states even at epsilon_norm=10, burst_k=25). The attack minimizes action margin without directional coherence — the pole self-corrects between steps.

**Proposed fix (not yet implemented):** Switch to a *targeted* attack that explicitly optimizes for the *opposite* of the correct action at each step, forcing directional consistency. This is the next task.
