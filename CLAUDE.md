# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Use `python3.8` (has all dependencies pre-installed). Run all scripts from the repo root.

```bash
pip install -r requirements.txt  # if setting up a new environment
```

Dependencies: `gymnasium`, `numpy`, `scipy`, `torch`, `stable-baselines3`.

---

## Method Overview

### Core Idea

At each step, choose between a **high-performance PPO policy** (higher return, vulnerable to adversarial attacks) and a **safe ATLA backup policy** (robust but lower return). A binary **SwitcherMLP** trained with Randomized Smoothing (RS) detects adversarial obs, and an **AnyTimeSwitcherController** manages transitions via a 4-phase state machine.

### Labeling (adversarial detection, not criticality)

- **y=0** (non-critical / clean): raw observation from PPO rollout
- **y=1** (critical / adversarial): `obs + tanh(attack_net(obs)) * eps` — the Zhang et al. optimal adversary applied to the same obs

50/50 split by construction. The switcher learns to detect adversarial perturbation.

### Randomized Smoothing Certification

`VanillaRSSwitcher.certify(obs)`:
1. Samples `n_samples` noisy copies: `obs + N(0, σ²I)` in ZFilter-normalized space
2. Counts majority vote → Clopper-Pearson lower bound `p_A_lower` at `confidence=0.001`
3. Returns `(pred, p_A_lower, R)` where `R = σ · Φ⁻¹(p_A_lower)`

Guarantee: if `pred==0` and any L2 perturbation `‖δ‖₂ ≤ R` is applied, the prediction remains class 0.

### 4-Phase AnyTimeSwitcherController

```
Phase 1 — PPO Monitoring (default state)
    Every step: RS-certify obs_ppo; use PPO for control
    "certified_safe" = (pred==0) AND (R >= delta_budget_l2)
    detection_k consecutive NOT-certified-safe steps → Phase 2

Phase 2 — ATLA Recovery  (recovery_k steps)
    Use ATLA; covers burst window + stabilization margin
    → Phase 3

Phase 3 — RS Commit Check  (at most commit_timeout_k steps)
    RS-certify; first step with (pred==0) AND (R >= delta) → Phase 4
    Forced commit after commit_timeout_k steps regardless
    → Phase 4

Phase 4 — Committed PPO  (permanent, rest of episode)
    PPO only; no further RS calls
    Justified by single-attack-per-episode threat model
```

**Why permanent commit (not loop)?** With a loop, Phase 1 restarts after recovery. P(false alarm per step) ≈ 10–40% depending on env → the controller spends most of the episode in ATLA via repeated false alarms. Permanent commit avoids this since the adversary fires at most once per episode.

**Dual ZFilter normalization**: Two independent ZFilters are maintained simultaneously:
- **PPO ZFilter** (from `Attack_PPO.model`'s `custom_env`): applied by `custom_env.step()` → `obs_ppo`; used for PPO inference and RS certification
- **ATLA ZFilter** (from `ATLA.model`'s `custom_env.state_filter`): applied to raw sim obs → `obs_atla`; used for ATLA inference

Per-step PPO↔ATLA switching is avoided in Phase 1 to prevent ZFilter churn (incompatible normalizations causing instability).

**Burst attack model**: attack starts at random T ~ U[0, t_candidate_max], lasts `burst_k` steps. PPO obs are perturbed; ATLA obs (from raw sim state) are always clean.

---

## CartPole Pipeline

### Workflow

```bash
# 1. Train performance PPO policy (10k steps — weak enough to fail under attack)
python3.8 scripts/train_perf.py --output models/perf_cartpole_ppo.zip --timesteps 10000

# 2. Build critical-state labels dataset
python3.8 scripts/build_labels.py \
  --perf-path models/perf_cartpole_weak \
  --dataset-out data/critical_dataset.npz \
  --epsilon-l2 0.5 \
  --burst-k 10 \
  --horizon-h 100 \
  --reward-drop-threshold 30 \
  --pgd-steps 10 \
  --n-attack-starts 3 \
  --episodes 8 \
  --subsample-every 4

# 3. Train binary switcher with RS noise augmentation
python3.8 scripts/train_switcher.py \
  --dataset data/critical_dataset.npz \
  --output models/switcher.pt \
  --hidden-dim 64 \
  --epochs 500 \
  --sigma 0.25

# 4. Evaluate under burst attacks
python3.8 scripts/evaluate_burst_attack.py \
  --perf-path models/perf_cartpole_weak \
  --switcher-path models/switcher.pt \
  --dataset data/critical_dataset.npz \
  --sigma 0.25 \
  --n-samples 1000 \
  --delta-budget-l2 0.5 \
  --epsilon-l2 0.5 \
  --burst-k 10 \
  --attack-mode fixed \
  --episodes 20
```

Pre-built artifacts:
- `data/critical_dataset.npz`
- `models/perf_cartpole_weak.zip` (10k steps — use this, not perf_cartpole_ppo)
- `models/switcher.pt`

**Note:** `stable-baselines3` appends `.zip` automatically — pass paths **without** `.zip` extension.

### Architecture (CartPole-specific)

CartPole uses **PGD L2 attacks** (not the Zhang et al. optimal adversary) for labeling. The backup is a **quantized LQR** (Riccati solution), not ATLA. Uses `CertifiedSwitcherController` (Phase 1 only — no 4-phase state machine).

### Key files

| File | Role |
|------|------|
| `cartpole_ags_rs_switcher/models.py` | `SwitcherMLP`: 1-hidden-layer binary classifier (obs → logit) |
| `cartpole_ags_rs_switcher/rs.py` | `VanillaRSSwitcher`: GPU-accelerated MC RS; `certify()` returns `(pred, p_A_lower, R_norm)` |
| `cartpole_ags_rs_switcher/evaluation.py` | Controller classes + `evaluate_controller()` |
| `cartpole_ags_rs_switcher/controllers.py` | `PerfPolicy` (PPO) and `QuantizedLQRBackup` |
| `cartpole_ags_rs_switcher/labeling.py` | `CriticalBurstLabeler`: PGD burst attacks to label critical/non-critical |
| `cartpole_ags_rs_switcher/attacks.py` | `pgd_l2_attack()`: targeted PGD in normalized obs space |
| `cartpole_ags_rs_switcher/training.py` | `train_switcher()`: BCE + noise augmentation |

### Key findings

- **Weak PPO (10k steps)**: clean return ~363, vulnerable to ε=0.5. Certified switcher: clean=500, attacked=500.
- **Strong PPO (20k+ steps)**: unbreakable — visits only equilibrium states, critical fraction = 0%.
- **Targeted PGD**: `attacks.py` uses `targeted=True` (default); minimizes `logit[clean] - logit[target]`.
- **Phase transition**: at ~19k steps PPO becomes fully robust; use 10k steps for meaningful experiments.
- **CartPole state_std** ≈ `[0.045, 0.136, 0.0055, 0.202]` (pole_angle std 25× smaller than cart_vel — all L2 quantities must be in normalized space).

---

## Hopper Pipeline

### Workflow

```bash
# 1. Build adversarial-detection dataset (clean → y=0, opt_attacked → y=1, 50/50 split)
python3.8 scripts/build_labels_hopper.py \
    --perf-path   Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --dataset-out data/hopper_critical_dataset.npz \
    --episodes 20 --subsample-every 5

# 2. Train binary switcher with RS noise augmentation
python3.8 scripts/train_switcher_hopper.py \
    --dataset data/hopper_critical_dataset.npz \
    --output models/hopper_switcher.pt \
    --hidden-dim 64 \
    --epochs 500 \
    --sigma 0.1

# 3. Evaluate 3 controllers under Zhang et al. attack (save JSON for plotting)
python3.8 scripts/evaluate_burst_attack_hopper.py \
    --perf-path   Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --backup-path Hopper/Hopper_ATLA.model \
    --switcher-path models/hopper_switcher.pt \
    --dataset  data/hopper_critical_dataset.npz \
    --sigma 0.1 --n-samples 10000 \
    --delta-budget-l2 0.075 \
    --episodes 30 --seed 0 \
    --burst-k 75 --t-candidate-max 100 \
    --recovery-k 100 --commit-timeout-k 5 \
    --device cuda \
    --output-json results/hopper_seed0.json

# 4. Generate plots (reads results/hopper_seed{0,42,123}.json)
python3.8 scripts/plot_hopper_results.py
```

Pre-built artifacts:
- `data/hopper_critical_dataset.npz` (7918 samples, 50% clean / 50% adv)
- `models/hopper_switcher.pt` (sigma=0.1, 94.1% accuracy)
- `results/hopper_seed{0,42,123}.json` (30 eps each, burst_k=75, t_max=100)
- `figures/hopper_*.pdf` (return comparison, fall rate, distribution)

### Architecture (Hopper-specific)

- `obs_dim=11`, `action_dim=3`, `eps=0.075` (L-inf), `name="Hopper"`
- uState: `qpos[6] + qvel[6] = 12D`; obs = normalized 11D via PPO ZFilter
- Raw 11D obs reconstructed via `_raw_obs_from_sim()`: `qpos[1:6] + clip(qvel[:6], -10, 10)`
- Termination: `height > 0.7 and |ang| < 0.2` (never fires for good PPO); hard `max_steps=1000` cap. "Fall" = `done=True` before step 1000.

### Key files

| File | Role |
|------|------|
| `hopper_ags_rs_switcher/controllers.py` | `HopperPerfPolicy` (PPO + Zhang attack) and `HopperBackupPolicy` (ATLA) |
| `hopper_ags_rs_switcher/evaluation.py` | `AlwaysPerfController`, `AlwaysBackupController`, `AnyTimeSwitcherController`, `evaluate_controller()` |
| `hopper_ags_rs_switcher/attacks.py` | `opt_attack()`: pre-trained adversary in normalized obs space |
| `hopper_ags_rs_switcher/labeling.py` | `HopperCriticalBurstLabeler`: detection dataset builder |
| `hopper_ags_rs_switcher/models.py` | Re-exports `SwitcherMLP` from CartPole module |
| `hopper_ags_rs_switcher/rs.py` | Re-exports `VanillaRSSwitcher` (GPU-accelerated) from CartPole module |
| `hopper_ags_rs_switcher/training.py` | `train_switcher()`: BCE + noise augmentation (used by all envs) |
| `scripts/plot_hopper_results.py` | Generates figures from multi-seed JSON results |

### Delta choice

`delta=0.075` = raw L-inf eps. Full L-2 budget = `0.075 * sqrt(11) ≈ 0.249`. Empirical `R_exec ≈ 0.30 > 0.249` — the commit gate certifies against the full L-2 budget. `delta=0.075` was chosen as the tightest achievable cert that also gives low false alarm rate with `detection_k=2`.

### AnyTimeSwitcherController parameters (Hopper)

| Param | Value | Rationale |
|---|---|---|
| `sigma` | 0.1 | Balances accuracy (94.1%) and cert radius R |
| `delta_budget_l2` | 0.075 | = eps; R_exec ≈ 0.30 >> 0.075 ✓ |
| `detection_k` | 2 | P(false alarm | clean)² ≈ 1%; fast detection |
| `recovery_k` | 100 | Covers burst_k=75 + 25-step stabilization |
| `commit_timeout_k` | 5 | Max steps in commit check before forced PPO |
| `n_samples` | 10000 | GPU-accelerated; tight Clopper-Pearson bounds |

**Why permanent commit?** With loop, P(false alarm in 1000 steps) ≈ 100% at δ=0.075 (R < δ ≈ 10% per step). Permanent commit is correct under single-attack-per-episode threat model.

**L-inf → L-2**: Zhang attack is L-inf(ε=0.075). For RS (which certifies L-2), full budget = `0.075√11 ≈ 0.249`. Empirical `R_exec ≈ 0.30 > 0.249` confirms cert covers full attack.

### Benchmark results (3 seeds × 30 episodes)
*(burst_k=75, t_candidate_max=100, recovery_k=100, sigma=0.1, delta=0.075, n_samples=10000, GPU)*

| Controller | Clean return | Attacked return | Falls (clean) | Falls (attacked) |
|---|:---:|:---:|:---:|:---:|
| Always PPO | 3613 ± 134 | 1954 ± 1427 | 4/90 | **53/90** |
| Always ATLA | 2359 ± 954 | 2466 ± 1006 | 69/90 | 68/90 |
| **Anytime Switcher** | **3569 ± 300** | **3183 ± 882** | **5/90** | **22/90** |

- **PPO collapses** under 75-step early burst (return halved, 53/90 falls)
- **Switcher is robust**: return barely changes (3569 → 3183); falls 5 → 22/90
- **Clean cost negligible**: Switcher ≈ PPO clean (3569 vs 3613, allow_perf=0.896)
- **ATLA alone unusable**: 69/90 falls even clean — cannot be standalone policy

### Checkpoint and path conventions

- **Attack checkpoint** (`Hopper_Attack_PPO.model`): contains `policy_model`, `adversary_policy_model`, `envs[0]`. All three MUST come from the same file for consistent ZFilter statistics.
- **ATLA checkpoint** (`Hopper_ATLA.model`): provides its own ZFilter via `custom_env.state_filter`.
- **custom_env API**: `reset(uState_12d, None, name="Hopper")` → normalized obs (11D); `step(action, change_filter=False, name="Hopper")` → `(result, norm_rew, is_done, info)` where `result[1]` is the next normalized obs.
- **old `policy_gradients/`**: renamed to `other_attacks/optimal_attack/opt_pg/` to avoid import conflict.
- **gym 0.26**: `custom_env.env.step()` returns 5-tuple; `_patch_gym_env()` converts to 4-tuple. `custom_env.env.reset()` may return `(obs, info)`; `start_episode()` unpacks this.

---

## HalfCheetah Pipeline

### Workflow

```bash
# 1. Build adversarial-detection dataset (clean obs → y=0, opt_attack(obs) → y=1)
python3.8 scripts/build_labels_halfcheetah.py \
    --perf-path   HalfCheetah/HalfCheetah_PPO.model \
    --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --dataset-out data/halfcheetah_critical_dataset.npz \
    --episodes 20 --subsample-every 5

# 2. Train binary switcher (sigma=0.2 for HalfCheetah)
python3.8 scripts/train_switcher_halfcheetah.py \
    --dataset data/halfcheetah_critical_dataset.npz \
    --output models/halfcheetah_switcher_s02.pt \
    --hidden-dim 64 --epochs 500 --sigma 0.2

# 3. Evaluate 3 controllers under Zhang et al. attack
python3.8 scripts/evaluate_burst_attack_halfcheetah.py \
    --perf-path   HalfCheetah/HalfCheetah_PPO.model \
    --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --backup-path HalfCheetah/HalfCheetah_ATLA.model \
    --switcher-path models/halfcheetah_switcher_s02.pt \
    --dataset  data/halfcheetah_critical_dataset.npz \
    --sigma 0.2 --n-samples 10000 \
    --delta-budget-l2 0.40 \
    --episodes 30 --seed 0 \
    --burst-k 100 --t-candidate-max 100 \
    --recovery-k 100 --commit-timeout-k 5 \
    --device cuda \
    --output-json results/halfcheetah_seed0.json

# 4. Generate plots (reads results/halfcheetah_seed{0,42,123}.json)
python3.8 scripts/plot_halfcheetah_results.py
```

Pre-built artifacts:
- `data/halfcheetah_critical_dataset.npz`
- `models/halfcheetah_switcher_s02.pt` (sigma=0.2, 98% accuracy)
- `results/halfcheetah_seed{0,42,123}.json` (30 eps each, burst_k=100, t_max=100)
- `figures/halfcheetah_*.pdf` (return comparison, distribution)

### Architecture (HalfCheetah-specific)

Same pipeline as Hopper: Zhang et al. optimal adversary, adversarial detection labeling, 4-phase `AnyTimeSwitcherController`.

- `obs_dim=17`, `action_dim=6`, `eps=0.15` (L-inf), `name="Cheetah"`
- uState: `qpos[9] + qvel[9] = 18D`; obs = ZFilter-normalized 17D
- Raw 17D obs: `qpos[1:9] + qvel[:9]` (8D + 9D, no velocity clipping)
- HalfCheetah **never terminates** (termination condition `ang > -0.8` rarely fires); story is return degradation, not falls. Hard `max_steps=1000` cap applied.

### Key files

| File | Role |
|------|------|
| `halfcheetah_ags_rs_switcher/controllers.py` | `CheetahPerfPolicy` (PPO + Zhang attack) and `CheetahBackupPolicy` (ATLA) |
| `halfcheetah_ags_rs_switcher/evaluation.py` | Same `AnyTimeSwitcherController` 4-phase design, HalfCheetah-specific env calls |
| `halfcheetah_ags_rs_switcher/attacks.py` | `opt_attack()` with eps=0.15 |
| `halfcheetah_ags_rs_switcher/labeling.py` | `CheetahCriticalBurstLabeler` |
| `halfcheetah_ags_rs_switcher/models.py` | Re-exports `SwitcherMLP` |
| `halfcheetah_ags_rs_switcher/rs.py` | Re-exports `VanillaRSSwitcher` |
| `halfcheetah_ags_rs_switcher/training.py` | Re-exports `train_switcher` from Hopper module |
| `scripts/plot_halfcheetah_results.py` | Generates figures from multi-seed JSON results |

### Delta choice rationale

Full L-inf → L-2: `0.15 * sqrt(17) ≈ 0.618`. **Unachievable**: post-ATLA states in PPO-normalized space are unusual (lower majority vote), limiting `R_exec ≈ 0.40–0.52` regardless of sigma.

**Chosen: sigma=0.2, delta=0.40.** Sweep confirmed R_exec ≈ 0.47 > delta=0.40 across all 3 seeds. The cert is real (R_exec > delta), just against a sub-budget attacker. Note: delta=0.40 > eps=0.15 (stronger than the raw L-inf budget in L-2 terms). Larger delta (0.45+) gave R_exec < delta (not certified).

### AnyTimeSwitcherController parameters (HalfCheetah)

| Param | Value | Rationale |
|---|---|---|
| `sigma` | 0.2 | Higher than Hopper; needed for R_exec to reach 0.40+ post-ATLA |
| `delta_budget_l2` | 0.40 | Max certified delta; R_exec ≈ 0.47 >> 0.40 ✓ |
| `detection_k` | 2 | Same as Hopper; HalfCheetah clean obs well-separated |
| `recovery_k` | 100 | Covers burst_k=100 exactly |
| `commit_timeout_k` | 5 | Max steps in commit check |
| `n_samples` | 10000 | GPU-accelerated |

### Benchmark results (3 seeds × 30 episodes)
*(burst_k=100, t_candidate_max=100, recovery_k=100, sigma=0.2, delta=0.40, n_samples=10000, GPU)*

| Controller | Clean return | Attacked return | allow_perf | R_exec |
|---|:---:|:---:|:---:|:---:|
| Always PPO | 7202 ± 340 | 5343 ± 1919 | 1.000 | — |
| Always ATLA | 5642 ± 45 | 5634 ± 42 | 0.000 | — |
| **Anytime Switcher** | **7001 ± 484** | **6159 ± 1608** | **0.898** | **≈0.47** |

- **PPO degrades** −26% under 100-step burst (7202 → 5343), high variance (σ=1919)
- **Switcher robust**: only −12% drop (7001 → 6159); R_exec ≈ 0.47 > delta=0.40 ✓ certified
- **Clean cost negligible**: 90% of steps on PPO; clean ≈ PPO (7001 vs 7202)
- **ATLA stable but capped**: immune to attack but cannot exploit clean states (~5640 always)
- **No falls** in any condition — story is return degradation

### Checkpoint and path conventions

- Models in `HalfCheetah/`: `HalfCheetah_PPO.model`, `HalfCheetah_Attack_PPO.model`, `HalfCheetah_ATLA.model`
- Same dual ZFilter and gym 0.26 compatibility as Hopper.
- Attack checkpoint and env must come from same file for consistent ZFilter statistics.

---

## Walker2D Pipeline (partial — in progress)

Walker2D has `obs_dim=17`, `action_dim=6`, `eps=0.05` (L-inf), `name="Walker2D"`. Models in `Walker2D/`.

Package `walker2d_ags_rs_switcher/` and scripts (`build_labels_walker2d.py`, `train_switcher_walker2d.py`, `evaluate_burst_attack_walker2d.py`) are implemented and functional.

**Known challenge**: Walker2D has a small attack budget (eps=0.05 → normalized L2 ≈ 0.20), which creates limited separation between clean and adversarial obs in normalized space. This causes low RS certified radius (mean R ≈ 0.08–0.13 for clean PPO obs), making the Phase 1 monitoring threshold hard to set without triggering frequent false alarms. Additionally, Walker2D's bipedal dynamics are sensitive to mid-episode controller switches, making any PPO→ATLA transition risky. The `AnyTimeSwitcherController` in `walker2d_ags_rs_switcher/evaluation.py` includes a `monitoring_delta` parameter (separate from `delta_budget_l2`) to allow pred-only detection in Phase 1 while maintaining the R threshold only for Phase 3 commit check.

Pre-built artifacts:
- `data/walker2d_critical_dataset.npz` (7838 samples, sigma=0.1, 90.4% accuracy)
- `models/walker2d_switcher.pt` (sigma=0.1)
- `models/walker2d_switcher_s02.pt` (sigma=0.2)

---

## Shared conventions

- **All L2 quantities** (`epsilon_l2`, `delta_budget_l2`, `sigma`, `R`) are in **ZFilter-normalized observation space**
- **`p1 = P(critical)`**: pred==0 → non-critical → use PPO; pred==1 → critical → use ATLA
- **Switcher checkpoint format**: `{"state_dict": ..., "obs_dim": int, "hidden_dim": int}`
- **Dataset `.npz` keys**: `X`, `y`, `state_mean`, `state_std`
- **`pos_weight = neg / pos`** in training: up-weights the critical class
- **Import path**: `other_attacks/optimal_attack/opt_pg/models.py` (renamed from `policy_gradients/` to avoid conflict)
- **Episode total reward**: read from `perf.custom_env.total_true_reward` after episode ends
