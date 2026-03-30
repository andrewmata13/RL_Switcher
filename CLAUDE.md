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

At each step, choose between a **high-performance PPO policy** (higher return, vulnerable to adversarial attacks) and a **safe ATLA backup policy** (robust but lower return). A binary **switcher** trained with Randomized Smoothing (RS) noise augmentation detects adversarial obs. Two controller variants manage transitions via a 4-phase state machine:
- **AnyTimeSwitcherController**: fixed-length ATLA recovery (recovery_k steps)
- **AdaptiveSwitcherController** (preferred): adaptive ATLA recovery -- monitors via RS during backup, exits when recovery_confirm_k consecutive certified-safe steps observed

Two certification backends:
- **VanillaRSSwitcher** (MC RS): `SwitcherMLP`, `SwitcherDeepMLP`, or `SwitcherRobustMLP`, statistical lower bound via sampling
- **GPSwitcher** (Gil-Pelaez): `SwitcherQuadMLP` with x^2+x activation, exact analytical certification, ~2-4ms constant time

**Proper certification**: `certified_safe = (pred==0) AND (R >= delta)`. Never use pred-only (`monitoring_delta=0.0`) — this discards the RS guarantee and gives empirical detection only. The guarantee requires `R >= delta` to hold.

### Labeling (adversarial detection, not criticality)

- **y=0** (non-critical / clean): raw observation from PPO rollout
- **y=1** (critical / adversarial): `obs + tanh(attack_net(obs)) * eps` -- the Zhang et al. optimal adversary applied to the same obs

50/50 split by construction. The switcher learns to detect adversarial perturbation.

### Randomized Smoothing Certification

`VanillaRSSwitcher.certify(obs)`:
1. Samples `n_samples` noisy copies: `obs + N(0, sigma^2 I)` in ZFilter-normalized space
2. Counts majority vote -> Clopper-Pearson lower bound `p_A_lower` at `confidence=0.001`
3. Returns `(pred, p_A_lower, R)` where `R = sigma * Phi^-1(p_A_lower)`

Guarantee: if `pred==0` and any L2 perturbation `||delta||_2 <= R` is applied, the prediction remains class 0.

### 4-Phase Controller State Machine

Both `AnyTimeSwitcherController` and `AdaptiveSwitcherController` share the same 4-phase structure. They differ only in Phase 2.

```
Phase 1 -- PPO Monitoring (default state)
    Every step: RS-certify obs_ppo; use PPO for control
    "certified_safe" = (pred==0) AND (R >= monitoring_delta)
    detection_k consecutive NOT-certified-safe steps -> Phase 2

Phase 2 -- ATLA Recovery
    AnyTime:  fixed recovery_k steps, no RS calls
    Adaptive: RS-certify each step; recovery_confirm_k consecutive
              certified-safe steps -> Phase 3 (no fixed cap)
    -> Phase 3

Phase 3 -- RS Commit Check  (at most commit_timeout_k steps)
    RS-certify; first step with (pred==0) AND (R >= delta_budget_l2) -> Phase 4
    Forced commit after commit_timeout_k steps regardless
    -> Phase 4

Phase 4 -- Committed PPO  (permanent, rest of episode)
    PPO only; no further RS calls
    Justified by single-attack-per-episode threat model
```

**Why permanent commit (not loop)?** With a loop, Phase 1 restarts after recovery. P(false alarm per step) ~ 10-40% depending on env -> the controller spends most of the episode in ATLA via repeated false alarms. Permanent commit avoids this since the adversary fires at most once per episode.

**Adaptive vs AnyTime**: The adaptive controller doesn't need to know burst_k in advance. During an active attack, adversarial obs consistently fail certification, so the controller naturally stays in ATLA. After the attack ends, it exits ATLA once the environment stabilizes (recovery_confirm_k consecutive safe steps). The AnyTime controller requires recovery_k >= burst_k to be set correctly.

**Dual ZFilter normalization**: Two independent ZFilters are maintained simultaneously:
- **PPO ZFilter** (from `Attack_PPO.model`'s `custom_env`): applied by `custom_env.step()` -> `obs_ppo`; used for PPO inference and RS certification
- **ATLA ZFilter** (from `ATLA.model`'s `custom_env.state_filter`): applied to raw sim obs -> `obs_atla`; used for ATLA inference

Per-step PPO<->ATLA switching is avoided in Phase 1 to prevent ZFilter churn (incompatible normalizations causing instability).

**Attack models**: `evaluate_controller()` supports three attack modes via `attack_mode`:
- `"single"` — one burst per episode (original). T ~ U[0, t_candidate_max], lasts `burst_k` steps.
- `"multi"` — `n_bursts` bursts of `burst_k` steps each, with at least `cooldown_k` gap between bursts.
- `"arbitrary"` — each step independently attacked with Bernoulli probability (expected total ≈ `burst_k` steps).

PPO obs are perturbed; ATLA obs (from raw sim state) are always clean. Supports both L-inf and L2 attacks via `attack_norm` parameter.

### ContinuousSwitcherController (arbitrary attacks)

For arbitrary (multi-burst, repeated) attacks, the 4-phase permanent-commit design breaks because the attacker can strike again after Phase 4. `ContinuousSwitcherController` uses hysteresis-based switching that loops indefinitely:

```
State: PPO (monitoring)
    RS-certify each step; track alarm_count.
    not certified safe -> alarm_count += 1
    certified safe     -> alarm_count = max(0, alarm_count - forgive_decay)
    alarm_count >= K_enter -> switch to ATLA

State: ATLA (recovery)
    RS-certify each step; track safe_count.
    certified safe     -> safe_count += 1
    not certified safe -> safe_count = 0
    safe_count >= K_exit -> switch to PPO
```

**Hysteresis parameters**:
- `K_enter`: consecutive alarm threshold to enter ATLA. Higher = more false alarm resistant, slower detection.
- `K_exit`: consecutive safe steps to return to PPO. Higher = more conservative recovery.
- `forgive_decay`: how fast alarm_count decays on safe steps. Higher = requires denser unsafe clusters to trigger.
- `monitoring_delta`: R threshold for Phase 1 detection (same as 4-phase controllers).

**Why hysteresis solves the false alarm problem**: With symmetric thresholds (K_enter = K_exit = 1, no forgive), the controller degrades to per-step switching — equivalent to looping Phase 1→2→1, spending most time in ATLA. Asymmetric thresholds with forgive_decay > 0 mean: scattered false alarms are forgiven and don't accumulate, but genuine attack bursts (where multiple consecutive steps fail certification) trigger ATLA entry. On the exit side, the controller stays in ATLA until the attack clearly stops.

**Per-step formal guarantee**: Each step the controller certifies. If pred==0 and R >= delta, the switcher prediction is provably robust within radius R regardless of attack timing. The hysteresis only governs the switching policy, not the per-step safety certificate.

---

## Code Organization

### Shared package: `rs_switcher_common/`

All shared infrastructure for MuJoCo environments lives in `rs_switcher_common/`:

| File | Role |
|------|------|
| `env_config.py` | `EnvConfig` dataclass + `HOPPER`/`HALFCHEETAH`/`WALKER2D` configs + `ENV_REGISTRY` |
| `models.py` | `SwitcherMLP` (1-hidden-layer), `SwitcherDeepMLP` (multi-layer ReLU), `SwitcherRobustMLP` (wide+BN+Dropout), `load_switcher()` |
| `rs.py` | `VanillaRSSwitcher`: GPU-accelerated MC RS; `certify()` returns `(pred, p_A_lower, R)` |
| `controllers.py` | `MuJoCoPerfPolicy` (PPO + Zhang attack), `MuJoCoBackupPolicy` (ATLA), `raw_obs_from_sim()` |
| `evaluation.py` | `AlwaysPerfController`, `AlwaysBackupController`, `AnyTimeSwitcherController`, `AdaptiveSwitcherController`, `evaluate_controller()` |
| `attacks.py` | `opt_attack()`: pre-trained adversary in normalized obs space (supports L-inf and L2 norms) |
| `labeling.py` | `CriticalBurstLabeler`: detection dataset builder; `collect_state_stats()` |
| `training.py` | `train_switcher()`: BCE + noise augmentation (supports `SwitcherMLP`, `SwitcherDeepMLP`, `SwitcherRobustMLP`); AdamW + cosine LR + n_noise_copies |
| `utils.py` | `set_seed`, `normalize`, `denormalize_eps` |
| `compat.py` | `ensure_paths()`, `patch_gym_env()` (gym 0.26 compatibility) |
| `gp_models.py` | `SwitcherQuadMLP`, `SwitcherQuadDeepMLP`, `SwitcherQuadSkipMLP`, `SwitcherBottleneckMLP`, `GPSwitcher` (Gil-Pelaez certifier), `load_gp_switcher()` |

### Gil-Pelaez (GP) Single-Pass Certification

`Single_Pass_Smoothing/` contains the GP certification library. `rs_switcher_common/gp_models.py` wraps it for the switcher:

- **`SwitcherQuadMLP`**: 2-class model with `Linear -> x^2+x -> Linear` architecture. The x^2+x activation enables exact analytical certification (no MC sampling).
- **`SwitcherQuadDeepMLP`**: Stacked `Linear+BN` backbone before x^2+x. At eval time, all Linear+BN layers fold into a single affine map via `fold_backbone()`, so certification still sees `Linear -> x^2+x -> Linear`. BN helps training optimization but doesn't increase eval-time capacity.
- **`SwitcherQuadSkipMLP`**: Parallel quad pathway (x²+x) + linear skip pathway. Architecture: `[W_quad @ x → x²+x; W_skip @ x] → cat → Linear(2)`. Margin under Gaussian noise is still generalized chi-squared — skip contribution adds to the linear coefficients in the eigendecomposed quadratic form without changing the structure. Use `certify_quad_skip_pA()`. Same speed as `SwitcherQuadMLP`. Accuracy ceiling: ~95.1% (same as pure quad — parallel linear features don't add capacity beyond x²+x's own +x term).
- **`SwitcherBottleneckMLP`**: `Linear(obs_dim, k) -> ReLU -> Linear(k, 2)`. Certified via k-dimensional Gauss-Hermite quadrature over the pre-activation Gaussian. Strictly more expressive than x²+x (piecewise-linear vs degree-2 polynomial), but hits the **quadrature curse-of-dimensionality**: cert cost scales as n^k. See "Architecture Comparison" table below.
- **`GPSwitcher`**: Drop-in replacement for `VanillaRSSwitcher`. Same `certify()` API returning `(pred, p_A_lower, R)`. Auto-selects n_quad for bottleneck models to stay within 200MB memory. Accepts `n_quad` kwarg override.
- **`load_gp_switcher(ckpt)`**: Reconstructs any GP model from checkpoint dict based on `model_type` key (`"quad"`, `"quad_deep"`, `"quad_skip"`, `"bottleneck"`).
- **Key advantage**: For binary classification (2 classes), GP has NO union bound penalty -- the certificate is exact.
- **Speed**: ~2-5ms/obs on CPU (constant, independent of sample count). RS scales linearly with n_samples.
- **Training**: Use `scripts/train_switcher_gp.py` with margin loss for better certified radii. Add `--arch bottleneck` for bottleneck model; `--skip-dim N` for skip pathway.

GP certification time per environment (CPU, h=512 for quad/skip):

| Env | quad/skip time | Fits 8ms budget |
|---|---|---|
| Hopper (11D) | ~1.8 ms | Yes |
| HalfCheetah (17D) | ~3-5 ms | Yes |
| Walker2D (17D) | ~3.3 ms | Yes |

**Architecture comparison (HalfCheetah, σ=0.2, 200 clean obs):**

| Model | Accuracy | Cert time | Mean R | frac R≥0.10 | frac R≥0.30 | Fits 8ms |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `SwitcherQuadMLP` h=512 | 95.1% | ~3ms | 0.324 | 94.5% | 53.0% | Yes |
| `SwitcherQuadSkipMLP` q=512 s=128 | 95.1% | ~5ms | 0.324 | 94.5% | 53.0% | Yes |
| `SwitcherBottleneckMLP` k=4 | 91.2% | ~2ms | 0.298 | 83.0% | 41.5% | Yes |
| `SwitcherBottleneckMLP` k=8 | 94.7% | ~14ms | — | — | — | **No** |

**Bottleneck finding**: x²+x and skip both ceiling at ~95.1%. Bottleneck k=4 is fast (2ms, n_quad=8) but weaker — fewer certified steps despite being piecewise-linear. k=8 approaches accuracy (94.7%) but 14ms exceeds the 8ms budget. The fundamental tradeoff: n^k quadrature cost grows exponentially, so you can't get both high capacity (large k) and fast certification. **Recommendation: use `SwitcherQuadSkipMLP` or `SwitcherQuadMLP` h=512 as the default GP certifier.**

### SwitcherRobustMLP (wide RS-certified classifier)

For MC RS certification, wider networks with BN and Dropout substantially improve accuracy, which directly translates to higher certified fractions (more steps where `R >= delta`).

**Architecture**: `[Linear(obs_dim, h) -> BN -> ReLU -> Dropout] * N -> Linear(h_last, 1)`
- Default: `hidden_dims=[1024, 1024, 512, 512, 256]`, `dropout=0.1`
- BN uses running stats at eval time → compatible with RS certification (deterministic output)
- Dropout disabled at eval → deterministic certification

**Training improvements** over `SwitcherMLP`/`SwitcherDeepMLP`:
- **AdamW** (weight decay in param update, not gradient) + **cosine LR schedule** (lr→lr*0.01)
- **n_noise_copies=4**: each sample expanded to K noisy copies per step, better approximates the smoothed classifier objective

```bash
# Train robust RS switcher
python3.8 scripts/train_switcher.py \
    --dataset data/halfcheetah_critical_dataset.npz \
    --output models/halfcheetah_switcher_robust.pt \
    --model-type robust --hidden-dims 1024,1024,512,512,256 --dropout 0.1 \
    --epochs 500 --sigma 0.2 --n-noise-copies 4 --lr 3e-4 --weight-decay 1e-4 \
    --lr-schedule cosine
```

**Checkpoint format**: `{"state_dict": ..., "obs_dim": int, "model_type": "robust", "hidden_dims": [...], "dropout": float}`
Use `load_switcher(ckpt)` from `rs_switcher_common/models.py` to reconstruct any switcher type.

**Sigma sweep insight**: The optimal certification sigma is NOT the training sigma. Certifying at `sigma_cert = sigma_train / 2` gives higher `p_A` per obs, which more than compensates for the smaller sigma in `R = sigma * Φ⁻¹(p_A)`. Example: Hopper at sigma_train=0.1, sigma_cert=0.05 → 88% of clean steps certify at R≥0.05 (vs 16.5% with GP at R≥0.075).

**Pre-built models**:
- `models/hopper_switcher_robust.pt` (sigma=0.1, 99.2% accuracy, hidden_dims=[1024,1024,512,512,256])
- `models/halfcheetah_switcher_robust.pt` (sigma=0.2, 99.9% accuracy)

### CartPole package: `cartpole_rs_switcher/`

CartPole-specific code (PGD attack, LQR backup, gymnasium-based eval):

| File | Role |
|------|------|
| `controllers.py` | `PerfPolicy` (SB3 PPO) and `QuantizedLQRBackup` |
| `evaluation.py` | `CertifiedSwitcherController` (Phase 1 only -- no 4-phase state machine) |
| `attacks.py` | `pgd_l2_attack()`: targeted PGD in normalized obs space |
| `labeling.py` | `CriticalBurstLabeler`: PGD burst attacks to label critical/non-critical |
| `config.py` | `LabelConfig`, `SwitcherTrainConfig`, `EvalConfig` dataclasses |
| `models.py`, `rs.py`, `training.py`, `utils.py` | Re-exports from `rs_switcher_common` |

### Environment configuration

`EnvConfig` in `rs_switcher_common/env_config.py` encodes per-env differences:

| Param | Hopper | HalfCheetah | Walker2D |
|-------|--------|-------------|----------|
| `name` | `"Hopper"` | `"Cheetah"` | `"Walker2D"` |
| `obs_dim` | 11 | 17 | 17 |
| `action_dim` | 3 | 6 | 6 |
| `eps` | 0.075 | 0.15 | 0.05 |
| `qpos_slice` | (1, 6) | (1, 9) | (1, 9) |
| `qvel_slice` | (0, 6) | (0, 9) | (0, 9) |
| `qvel_clip` | 10.0 | None | 10.0 |

---

## CartPole Pipeline

### Workflow

```bash
# 1. Train performance PPO policy (10k steps -- weak enough to fail under attack)
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
- `models/perf_cartpole_weak.zip` (10k steps -- use this, not perf_cartpole_ppo)
- `models/switcher.pt`

**Note:** `stable-baselines3` appends `.zip` automatically -- pass paths **without** `.zip` extension.

### Architecture (CartPole-specific)

CartPole uses **PGD L2 attacks** (not the Zhang et al. optimal adversary) for labeling. The backup is a **quantized LQR** (Riccati solution), not ATLA. Uses `CertifiedSwitcherController` (Phase 1 only -- no 4-phase state machine).

### Key findings

- **Weak PPO (10k steps)**: clean return ~363, vulnerable to eps=0.5. Certified switcher: clean=500, attacked=500.
- **Strong PPO (20k+ steps)**: unbreakable -- visits only equilibrium states, critical fraction = 0%.
- **Targeted PGD**: `attacks.py` uses `targeted=True` (default); minimizes `logit[clean] - logit[target]`.
- **Phase transition**: at ~19k steps PPO becomes fully robust; use 10k steps for meaningful experiments.
- **CartPole state_std** ~ `[0.045, 0.136, 0.0055, 0.202]` (pole_angle std 25x smaller than cart_vel -- all L2 quantities must be in normalized space).

---

## MuJoCo Pipeline (Hopper, HalfCheetah, Walker2D)

All MuJoCo environments use the same unified scripts with `--env` parameter.

### Workflow

```bash
# 1. Build adversarial-detection dataset (clean -> y=0, opt_attacked -> y=1, 50/50 split)
python3.8 scripts/build_labels_mujoco.py --env hopper \
    --perf-path   Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --dataset-out data/hopper_critical_dataset.npz \
    --episodes 20 --subsample-every 5

# 2. Train GP switcher (preferred -- exact certification)
python3.8 scripts/train_switcher_gp.py \
    --dataset data/hopper_critical_dataset.npz \
    --output models/hopper_switcher_gp_h512.pt \
    --hidden-dim 512 --epochs 500 --sigma 0.1

# 3. Evaluate with adaptive controller under L2 attack
python3.8 scripts/evaluate_burst_attack_gp.py --env hopper \
    --perf-path   Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --backup-path Hopper/Hopper_ATLA.model \
    --gp-switcher-path models/hopper_switcher_gp_h512.pt \
    --dataset data/hopper_critical_dataset.npz \
    --sigma 0.1 --delta-budget-l2 0.075 \
    --episodes 30 --seed 0 --burst-k 75 --t-candidate-max 100 \
    --recovery-confirm-k 10 --commit-timeout-k 5 \
    --attack-norm l2 --attack-eps 0.13 \
    --output-json results/hopper_adaptive_l2.json

# 4. Worst-case T* evaluation (finds hardest attack start time)
python3.8 scripts/evaluate_worst_case_start.py --env hopper \
    --perf-path   Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --backup-path Hopper/Hopper_ATLA.model \
    --switcher-path models/hopper_switcher.pt \
    --dataset data/hopper_critical_dataset.npz \
    --seeds 30 --device cpu
```

### Hopper

- `obs_dim=11`, `action_dim=3`, `eps=0.075` (L-inf), `name="Hopper"`
- uState: `qpos[6] + qvel[6] = 12D`; obs = normalized 11D via PPO ZFilter
- Raw 11D obs: `qpos[1:6] + clip(qvel[:6], -10, 10)`
- Termination: `height > 0.7 and |ang| < 0.2` (never fires for good PPO); hard `max_steps=1000` cap. "Fall" = `done=True` before step 1000.
- Pre-built: `data/hopper_critical_dataset.npz` (7918 samples)
- GP model: `models/hopper_switcher_gp_h512.pt` (sigma=0.1, ~85% accuracy, hidden_dim=512)
- RS model (small): `models/hopper_switcher.pt` (sigma=0.1, 94.1% accuracy, hidden_dim=64)
- RS model (robust): `models/hopper_switcher_robust.pt` (sigma=0.1, 99.2% accuracy, hidden_dims=[1024,1024,512,512,256])
- **Hopper bottleneck**: `Hopper_Attack_ATLA.model` (correct backup) falls 8/10 as always_backup → continuous controller unusable until better ATLA trained. Use HalfCheetah for continuous controller experiments.

**Adaptive GP results (seed 0, 30 episodes, L2 attack eps=0.13, burst_k=75):**

| Controller | Clean return | Attacked return | Falls (clean) | Falls (attacked) |
|---|:---:|:---:|:---:|:---:|
| Always PPO | 3614 +/- 59 | 2572 +/- 1380 | 1/30 | **12/30** |
| Always ATLA | 2694 +/- 948 | 2375 +/- 886 | 20/30 | 24/30 |
| **Adaptive GP** | **3601 +/- 147** | **3474 +/- 410** | **2/30** | **5/30** |

| Param | Value | Rationale |
|---|---|---|
| `sigma` | 0.1 | Balances accuracy (~85% GP) and cert radius |
| `delta_budget_l2` | 0.075 | Controller detection/commit threshold |
| `L2 attack eps` | 0.13 | Causes meaningful degradation (PPO: 3614->2572) |
| `detection_k` | 2 | Fast detection |
| `recovery_confirm_k` | 10 | Adaptive exit from ATLA |
| `commit_timeout_k` | 5 | Max steps in commit check before forced PPO |

### HalfCheetah

- `obs_dim=17`, `action_dim=6`, `eps=0.15` (L-inf), `name="Cheetah"`
- uState: `qpos[9] + qvel[9] = 18D`; obs = ZFilter-normalized 17D
- Raw 17D obs: `qpos[1:9] + qvel[:9]` (8D + 9D, no velocity clipping)
- HalfCheetah **never terminates**; story is return degradation, not falls. Hard `max_steps=1000` cap.
- Pre-built: `data/halfcheetah_critical_dataset.npz`
- GP model: `models/halfcheetah_switcher_gp_s02.pt` (sigma=0.2, 95.2% accuracy, hidden_dim=512)
- RS model: `models/halfcheetah_switcher_robust.pt` (sigma=0.2, 99.9% accuracy, hidden_dims=[1024,1024,512,512,256])

**Adaptive GP results (seed 0, 30 episodes, L2 attack eps=0.50, burst_k=100):**

| Controller | Clean return | Attacked return |
|---|:---:|:---:|
| Always PPO | 7267 +/- 108 | 5728 +/- 1822 |
| Always ATLA | 5638 +/- 48 | 5624 +/- 45 |
| **Adaptive GP** | **6931 +/- 1159** | **6445 +/- 1354** |

| Param | Value |
|---|---|
| `sigma` | 0.2 |
| `delta_budget_l2` | 0.30 |
| `L2 attack eps` | 0.50 |
| `detection_k` | 5 |
| `recovery_confirm_k` | 3 |

**Continuous RS results (seed 0, 10 episodes, L2 attack eps=0.50, multi-burst 3×100 with cooldown=100):**

| Controller | Clean return | Attacked return | PPO% clean | PPO% attacked |
|---|:---:|:---:|:---:|:---:|
| Always PPO | 7287 +/- 94 | 2525 +/- 1477 | 100% | 100% |
| Always ATLA | 5625 +/- 30 | 5659 +/- 40 | 0% | 0% |
| Adaptive RS | 5818 +/- 343 | 5672 +/- 158 | 11.4% | 6.1% |
| **Continuous RS** | **7170 +/- 94** | **6421 +/- 119** | **96.8%** | **56.7%** |

Key: HalfCheetah is ideal for ContinuousSwitcherController — never terminates, so repeated ATLA periods cause degradation not falls. At sigma=0.10/delta=0.10, 99.5% of clean steps certify (0.5% false alarm rate), enabling K_enter=3, K_exit=5 without excessive ATLA time.

```bash
# Evaluate continuous RS controller on HalfCheetah
python3.8 scripts/evaluate_continuous_controller.py --env halfcheetah \
    --perf-path HalfCheetah/HalfCheetah_PPO.model \
    --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --backup-path HalfCheetah/HalfCheetah_ATLA.model \
    --rs-switcher-path models/halfcheetah_switcher_robust.pt \
    --dataset data/halfcheetah_critical_dataset.npz \
    --sigma 0.1 --delta-budget-l2 0.1 \
    --n-samples 300 --episodes 10 --seed 0 \
    --attack-mode multi --n-bursts 3 --burst-k 100 --cooldown-k 100 \
    --K-enter 3 --K-exit 5 \
    --attack-norm l2 --attack-eps 0.5 \
    --output-json results/halfcheetah_continuous_rs.json
```

### Walker2D

- `obs_dim=17`, `action_dim=6`, `eps=0.05` (L-inf), `name="Walker2D"`. Models in `Walker2D/`.
- Pre-built: `data/walker2d_critical_dataset.npz` (7838 samples)
- GP model: `models/walker2d_switcher_gp.pt` (sigma=0.1, 79% accuracy, hidden_dim=512)

**Adaptive GP results (seed 0, 30 episodes, L2 attack eps=0.10, burst_k=75):**

| Controller | Clean return | Attacked return | Falls (clean) | Falls (attacked) |
|---|:---:|:---:|:---:|:---:|
| Always PPO | 4590 +/- 222 | 3996 +/- 1281 | 1/30 | 7/30 |
| Always ATLA | 3606 +/- 758 | 3524 +/- 886 | 8/30 | 9/30 |
| **Adaptive GP** | **4172 +/- 1116** | **4327 +/- 937** | **4/30** | **2/30** |

| Param | Value |
|---|---|
| `sigma` | 0.1 |
| `delta_budget_l2` | 0.05 |
| `monitoring_delta` | 0.0 (**pred-only — bypasses formal RS guarantee**; empirical detection only) |
| `L2 attack eps` | 0.10 |
| `detection_k` | 2 |
| `recovery_confirm_k` | 5 |

---

## Checkpoint and path conventions

- **Attack checkpoint** (e.g. `Hopper_Attack_PPO.model`): contains `policy_model`, `adversary_policy_model`, `envs[0]`. All three MUST come from the same file for consistent ZFilter statistics.
- **ATLA checkpoint** (e.g. `Hopper_Attack_ATLA.model` — use the `Attack_ATLA` file, not bare `ATLA.model`): provides its own ZFilter via `custom_env.state_filter`.
- **custom_env API**: `reset(uState, None, name=...)` -> normalized obs; `step(action, change_filter=False, name=...)` -> `(result, norm_rew, is_done, info)` where `result[1]` is the next normalized obs.
- **`policy_gradients/`**: required at repo root for `torch.load` unpickling of `.model` checkpoints. Do not delete.
- **gym 0.26**: `patch_gym_env()` in `rs_switcher_common/compat.py` converts 5-tuple step returns to 4-tuple.

## Shared conventions

- **All L2 quantities** (`epsilon_l2`, `delta_budget_l2`, `sigma`, `R`) are in **ZFilter-normalized observation space**
- **`p1 = P(critical)`**: pred==0 -> non-critical -> use PPO; pred==1 -> critical -> use ATLA
- **Switcher checkpoint format**: `{"state_dict": ..., "obs_dim": int, "hidden_dim": int}` (may include `hidden_dims` for deep models, `model_type`/`backbone_dims` for GP models, `dropout` for robust models). Use `load_switcher(ckpt)` to reconstruct any type.
- **Dataset `.npz` keys**: `X`, `y`, `state_mean`, `state_std`
- **`pos_weight = neg / pos`** in training: up-weights the critical class
- **Import path**: `other_attacks/optimal_attack/opt_pg/models.py` (renamed from `policy_gradients/` to avoid conflict)
- **Episode total reward**: read from `perf.custom_env.total_true_reward` after episode ends

---

## Open Problems and Future Work

### Certification radius vs attack strength gap

The fundamental tension: certified radii are small, and at those small L2 budgets the attack barely hurts. There's a gap between what we can certify and what actually degrades the policy.

**Current state** (GP certification, clean observations):

| Env | sigma | Avg certified R | L2 eps that degrades PPO | % clean obs with R >= eps |
|---|---|---|---|---|
| Hopper | 0.1 | 0.079 | 0.13 | 16.5% |
| HalfCheetah | 0.2 | 0.323 | 0.50 | 14.5% |
| Walker2D | 0.1 | 0.090 | 0.10 | 36.0% |

**With SwitcherRobustMLP + smaller cert sigma** (99%+ accuracy, n_samples=300-500):

| Env | train sigma | cert sigma | cert delta | % clean obs certified |
|---|---|---|---|---|
| Hopper | 0.1 | 0.05 | 0.05 | ~88% |
| HalfCheetah | 0.2 | 0.10 | 0.10 | ~99.5% |

Key insight: certify at `sigma_cert = sigma_train / 2`. Higher accuracy → p_A close to 1 → R = sigma*Φ⁻¹(p_A) stays large even with smaller sigma. The certified fraction matters more than avg R for controller usability.

The certificate says: "if R >= delta, then no perturbation within delta can flip the prediction." For full formal guarantees, we need `delta = L2_eps` (the attacker's budget). But with `delta = L2_eps`, most clean observations fail certification (84-86% false alarm rate for GP), making the controller unusable.

**Current workaround**: set `delta < L2_eps`. Detection works empirically (the switcher correctly identifies adversarial obs) but the formal RS guarantee doesn't fully cover the attack budget. The practical results are strong -- the adaptive controller significantly reduces attack damage across all envs.

**Possible solutions to close the gap**:

1. **Train at higher sigma**: R = sigma * Phi^-1(pA). Higher sigma training increases noise robustness, yielding larger R. Tested up to sigma=0.3 on Hopper -- accuracy drops (~71% GP at sigma=0.3 vs ~85% at sigma=0.1) but radii increase. Diminishing returns beyond sigma=0.3 for the x^2+x architecture.

2. **Stronger architectures for GP** (EXHAUSTED): The x^2+x activation limits the model to degree-2 polynomials. All variants tested:
   - `SwitcherQuadDeepMLP` (stacked Linear+BN): BN folds into a single affine map at eval — no capacity gain.
   - `SwitcherQuadSkipMLP` (parallel linear skip): skip adds linear features but they're already captured by the +x term in x²+x — same 95.1% ceiling.
   - `SwitcherBottleneckMLP` (ReLU, k-dim quadrature): more expressive (piecewise-linear), certified via Gauss-Hermite. k=4 fits budget (2ms) but achieves only 91.2% and lower certified fractions. k=8 gets 94.7% but 14ms is too slow. The n^k quadrature cost is the fundamental barrier. No known certifiable architecture beats x²+x at h=512 within the 8ms budget.

3. **Retrain the adversary in L2** (IN PROGRESS): See "L2 Native Adversary" section below. The Zhang et al. `agent.py` has been modified to support L2 norm projection, and `scripts/train_l2_adversary.py` provides both adversary-only and minimax ATLA training. Preliminary Hopper results show the native L2 adversary is highly effective.

4. **Tighter certification methods**: GP is exact for the x^2+x architecture but the architecture itself limits radii. Explore certification methods for more expressive models (e.g., Lipschitz-bounded networks, interval bound propagation).

5. **Accept the gap for practical use**: The empirical detection works. The formal guarantee covers perturbations up to the certified radius; larger perturbations are detected with high probability but without formal guarantee.

---

## L2 Native Adversary (In Progress)

### Motivation

The existing adversaries are trained for L-inf attacks. When evaluated under L2 norm, the perturbation direction is suboptimal (post-hoc projection of L-inf directions onto L2 ball). A natively L2-trained adversary finds more damaging L2 perturbation directions, causing greater PPO degradation at the same L2 budget. This shrinks the gap between what hurts the policy and what can be certified.

### Changes to Zhang et al. framework

`other_attacks/optimal_attack/opt_pg/agent.py` — three perturbation projection sites (trajectory collection, perturbed state collection, `apply_attack`) modified to support L2 norm via `getattr(self.params, 'ADV_NORM', 'linf')`. When `adv_norm == 'l2'`, replaces per-dimension hardtanh+bounds clamping with L2 ball projection:
```python
p_norm = perturbation.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
perturbation = perturbation / p_norm * float(self.ADV_EPS)
```

`config.json` — added `"adv_norm": "linf"` (default preserves backward compatibility).
`run.py` — added `--adv-norm` CLI argument.

### Training script: `scripts/train_l2_adversary.py`

Standalone PPO-based trainer (bypasses the full Zhang et al. Trainer which has environment wrapper incompatibilities for Hopper). Two modes:

| Mode | What trains | Use case |
|------|-------------|----------|
| `--mode adv-only` | L2 adversary against **frozen** PPO | Step 1: find L2 attack directions |
| `--mode minimax` | ATLA policy + L2 adversary co-train | Step 2: make ATLA robust to L2 attacks |

Key features:
- Uses **true reward** (via `total_true_reward` differencing), not normalized reward. The normalized reward is ~100x weaker and prevents adversary convergence.
- `--warm-start` flag for minimax mode: loads existing ATLA weights for faster convergence.
- Saves in standard checkpoint format (compatible with `MuJoCoPerfPolicy.load`).

### Training budget finding

The L-inf training uses per-dimension budget `(bounds_range) * eps = 10 * 0.075 = 0.75`, giving effective L2 norm up to ~2.49. Training an L2 adversary at the evaluation budget (eps=0.13) is far too small -- the adversary cannot learn (PPO return stays at ~3618). Training at eps=0.5 produces a strong adversary.

### Preliminary Hopper results (L2 adversary trained at eps=0.5)

**PPO return under L2 attack at various eval budgets (30 episodes, clean baseline: 3614):**

| L2 eval eps | L-inf projected | Native L2 | Avg certified R |
|:-----------:|:---------------:|:---------:|:---------------:|
| 0.05 | — | 3626 | 0.079 |
| 0.10 | — | **1050** | 0.079 |
| 0.13 | 479 | 751 | 0.079 |
| 0.20 | 382 | 457 | 0.079 |

Key findings:
- Native L2 adversary causes massive damage at eps=0.10 (3614 -> 1050), close to the certifiable radius of 0.079.
- At eps=0.13, the L-inf projected adversary is actually more damaging (479 vs 751) because its directions were optimized at a larger training budget. Both cause 100% falls.
- The gap between certifiable R (0.079) and "L2 eps that hurts" shrinks from 0.13 to ~0.075 with the native L2 adversary.

### Remaining steps

1. **Train ATLA with L2 adversary (minimax)**: Use `--mode minimax --warm-start Hopper/Hopper_ATLA.model` for fast convergence. Verify clean return ≈ 2694 (current ATLA).
2. **Rebuild detection dataset**: `build_labels_mujoco.py --attack-path Hopper/Hopper_Attack_PPO_L2.model`
3. **Retrain switcher**: `train_switcher_gp.py` on new L2-based dataset.
4. **Evaluate full adaptive controller**: `evaluate_burst_attack_gp.py --attack-norm l2` with L2-ATLA, L2 adversary, L2 switcher.
5. **Extend to HalfCheetah and Walker2D**.

### GPU commands (ready to run)

```bash
# Train robust RS switchers (all envs, ~10 min each on GPU)
python3.8 scripts/train_switcher.py \
    --dataset data/walker2d_critical_dataset.npz \
    --output models/walker2d_switcher_robust.pt \
    --model-type robust --hidden-dims 1024,1024,512,512,256 --dropout 0.1 \
    --epochs 500 --sigma 0.1 --n-noise-copies 4 --lr 3e-4 --weight-decay 1e-4 \
    --lr-schedule cosine

# Evaluate continuous controller (Walker2D, 30 episodes)
python3.8 scripts/evaluate_continuous_controller.py --env walker2d \
    --perf-path Walker2D/Walker2D_PPO.model \
    --attack-path Walker2D/Walker2D_Attack_PPO.model \
    --backup-path Walker2D/Walker2D_ATLA.model \
    --rs-switcher-path models/walker2d_switcher_robust.pt \
    --dataset data/walker2d_critical_dataset.npz \
    --sigma 0.05 --delta-budget-l2 0.05 \
    --n-samples 1000 --episodes 30 --seed 0 \
    --attack-mode multi --n-bursts 3 --burst-k 75 --cooldown-k 75 \
    --K-enter 3 --K-exit 10 \
    --attack-norm l2 --attack-eps 0.10

# Run K_enter/K_exit sweep on HalfCheetah (multi-burst vs arbitrary attack)
for K_enter in 2 3 5; do
for K_exit in 3 5 10 15; do
for mode in multi arbitrary; do
python3.8 scripts/evaluate_continuous_controller.py --env halfcheetah \
    --perf-path HalfCheetah/HalfCheetah_PPO.model \
    --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --backup-path HalfCheetah/HalfCheetah_ATLA.model \
    --rs-switcher-path models/halfcheetah_switcher_robust.pt \
    --dataset data/halfcheetah_critical_dataset.npz \
    --sigma 0.1 --delta-budget-l2 0.1 --n-samples 300 --episodes 30 --seed 0 \
    --attack-mode $mode --n-bursts 3 --burst-k 100 --cooldown-k 100 \
    --K-enter $K_enter --K-exit $K_exit \
    --attack-norm l2 --attack-eps 0.5 \
    --output-json results/halfcheetah_cont_ke${K_enter}_kx${K_exit}_${mode}.json
done; done; done
```

```bash
# Step 1: L2 adversary (already done for Hopper, ~40 min CPU)
python3.8 scripts/train_l2_adversary.py --mode adv-only --env hopper \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --eps 0.5 --train-steps 500 --rollout-len 2048 \
    --lr 3e-4 --val-lr 1e-3 --entropy-coeff 0.001 \
    --seed 0 --output Hopper/Hopper_Attack_PPO_L2.model

# Step 2: ATLA minimax (warm-start, ~2-4 hours CPU)
python3.8 scripts/train_l2_adversary.py --mode minimax --env hopper \
    --attack-path Hopper/Hopper_Attack_PPO.model \
    --warm-start Hopper/Hopper_ATLA.model \
    --eps 0.5 --train-steps 500 --rollout-len 2048 \
    --lr 3e-4 --adv-lr 3e-5 --seed 0 \
    --output Hopper/Hopper_ATLA_L2.model

# Step 3: Rebuild dataset
python3.8 scripts/build_labels_mujoco.py --env hopper \
    --perf-path Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO_L2.model \
    --dataset-out data/hopper_l2_critical_dataset.npz \
    --episodes 20 --subsample-every 5

# Step 4: Retrain GP switcher
python3.8 scripts/train_switcher_gp.py \
    --dataset data/hopper_l2_critical_dataset.npz \
    --output models/hopper_switcher_gp_l2.pt \
    --hidden-dim 512 --epochs 500 --sigma 0.1

# Step 5: Evaluate adaptive controller
python3.8 scripts/evaluate_burst_attack_gp.py --env hopper \
    --perf-path Hopper/Hopper_PPO.model \
    --attack-path Hopper/Hopper_Attack_PPO_L2.model \
    --backup-path Hopper/Hopper_ATLA_L2.model \
    --gp-switcher-path models/hopper_switcher_gp_l2.pt \
    --dataset data/hopper_l2_critical_dataset.npz \
    --sigma 0.1 --delta-budget-l2 0.075 \
    --episodes 30 --seed 0 --burst-k 75 --t-candidate-max 100 \
    --recovery-confirm-k 10 --commit-timeout-k 5 \
    --attack-norm l2 --attack-eps 0.10 \
    --output-json results/hopper_adaptive_l2_native.json
```
