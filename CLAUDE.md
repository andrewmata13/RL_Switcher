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
- **VanillaRSSwitcher** (MC RS): `SwitcherMLP` or `SwitcherDeepMLP`, statistical lower bound via sampling
- **GPSwitcher** (Gil-Pelaez): `SwitcherQuadMLP` with x^2+x activation, exact analytical certification, ~2-4ms constant time

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

**Burst attack model**: attack starts at random T ~ U[0, t_candidate_max] (or fixed T via `t_candidate_fixed`), lasts `burst_k` steps. PPO obs are perturbed; ATLA obs (from raw sim state) are always clean. Supports both L-inf and L2 attacks via `attack_norm` parameter.

---

## Code Organization

### Shared package: `rs_switcher_common/`

All shared infrastructure for MuJoCo environments lives in `rs_switcher_common/`:

| File | Role |
|------|------|
| `env_config.py` | `EnvConfig` dataclass + `HOPPER`/`HALFCHEETAH`/`WALKER2D` configs + `ENV_REGISTRY` |
| `models.py` | `SwitcherMLP` (1-hidden-layer) and `SwitcherDeepMLP` (multi-layer ReLU) binary classifiers |
| `rs.py` | `VanillaRSSwitcher`: GPU-accelerated MC RS; `certify()` returns `(pred, p_A_lower, R)` |
| `controllers.py` | `MuJoCoPerfPolicy` (PPO + Zhang attack), `MuJoCoBackupPolicy` (ATLA), `raw_obs_from_sim()` |
| `evaluation.py` | `AlwaysPerfController`, `AlwaysBackupController`, `AnyTimeSwitcherController`, `AdaptiveSwitcherController`, `evaluate_controller()` |
| `attacks.py` | `opt_attack()`: pre-trained adversary in normalized obs space (supports L-inf and L2 norms) |
| `labeling.py` | `CriticalBurstLabeler`: detection dataset builder; `collect_state_stats()` |
| `training.py` | `train_switcher()`: BCE + noise augmentation (supports `SwitcherMLP` and `SwitcherDeepMLP`) |
| `utils.py` | `set_seed`, `normalize`, `denormalize_eps` |
| `compat.py` | `ensure_paths()`, `patch_gym_env()` (gym 0.26 compatibility) |
| `gp_models.py` | `SwitcherQuadMLP` (2-class x^2+x model), `SwitcherQuadDeepMLP` (stacked Linear+BN backbone), `GPSwitcher` (Gil-Pelaez certifier) |

### Gil-Pelaez (GP) Single-Pass Certification

`Single_Pass_Smoothing/` contains the GP certification library. `rs_switcher_common/gp_models.py` wraps it for the switcher:

- **`SwitcherQuadMLP`**: 2-class model with `Linear -> x^2+x -> Linear` architecture. The x^2+x activation enables exact analytical certification (no MC sampling).
- **`SwitcherQuadDeepMLP`**: Stacked `Linear+BN` backbone before x^2+x. At eval time, all Linear+BN layers fold into a single affine map via `fold_backbone()`, so certification still sees `Linear -> x^2+x -> Linear`. BN helps training optimization but doesn't increase eval-time capacity.
- **`GPSwitcher`**: Drop-in replacement for `VanillaRSSwitcher`. Same `certify()` API returning `(pred, p_A_lower, R)`.
- **Key advantage**: For binary classification (2 classes), GP has NO union bound penalty -- the certificate is exact.
- **Speed**: ~2-4ms/obs on CPU (constant, independent of sample count). RS scales linearly with n_samples.
- **Training**: Use `scripts/train_switcher_gp.py` with margin loss for better certified radii.

GP certification time per environment (CPU, h=512):

| Env | GP time | Fits 8ms control budget |
|---|---|---|
| Hopper (11D) | ~1.8 ms | Yes |
| HalfCheetah (17D) | ~2.9 ms | Yes |
| Walker2D (17D) | ~3.3 ms | Yes |

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
- RS model: `models/hopper_switcher.pt` (sigma=0.1, 94.1% accuracy, hidden_dim=64)

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
| `monitoring_delta` | 0.0 (pred-only detection in Phase 1) |
| `L2 attack eps` | 0.10 |
| `detection_k` | 2 |
| `recovery_confirm_k` | 5 |

---

## Checkpoint and path conventions

- **Attack checkpoint** (e.g. `Hopper_Attack_PPO.model`): contains `policy_model`, `adversary_policy_model`, `envs[0]`. All three MUST come from the same file for consistent ZFilter statistics.
- **ATLA checkpoint** (e.g. `Hopper_ATLA.model`): provides its own ZFilter via `custom_env.state_filter`.
- **custom_env API**: `reset(uState, None, name=...)` -> normalized obs; `step(action, change_filter=False, name=...)` -> `(result, norm_rew, is_done, info)` where `result[1]` is the next normalized obs.
- **`policy_gradients/`**: required at repo root for `torch.load` unpickling of `.model` checkpoints. Do not delete.
- **gym 0.26**: `patch_gym_env()` in `rs_switcher_common/compat.py` converts 5-tuple step returns to 4-tuple.

## Shared conventions

- **All L2 quantities** (`epsilon_l2`, `delta_budget_l2`, `sigma`, `R`) are in **ZFilter-normalized observation space**
- **`p1 = P(critical)`**: pred==0 -> non-critical -> use PPO; pred==1 -> critical -> use ATLA
- **Switcher checkpoint format**: `{"state_dict": ..., "obs_dim": int, "hidden_dim": int}` (may include `hidden_dims` for deep models, `model_type`/`backbone_dims` for GP models)
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

The certificate says: "if R >= delta, then no perturbation within delta can flip the prediction." For full formal guarantees, we need `delta = L2_eps` (the attacker's budget). But with `delta = L2_eps`, most clean observations fail certification (84-86% false alarm rate), making the controller unusable.

**Current workaround**: set `delta < L2_eps`. Detection works empirically (the switcher correctly identifies adversarial obs) but the formal RS guarantee doesn't fully cover the attack budget. The practical results are strong -- the adaptive controller significantly reduces attack damage across all envs.

**Possible solutions to close the gap**:

1. **Train at higher sigma**: R = sigma * Phi^-1(pA). Higher sigma training increases noise robustness, yielding larger R. Tested up to sigma=0.3 on Hopper -- accuracy drops (~71% GP at sigma=0.3 vs ~85% at sigma=0.1) but radii increase. Diminishing returns beyond sigma=0.3 for the x^2+x architecture.

2. **Stronger architectures for GP**: The x^2+x activation limits the model to degree-2 polynomials. `SwitcherQuadDeepMLP` (stacked Linear+BN backbone) was tested but BN layers fold into a single linear map at eval time, so it doesn't increase capacity. Need architectures that are both more expressive AND analytically certifiable.

3. **Retrain the adversary in L2**: Current adversary is trained for L-inf attacks. Post-hoc L2 projection (normalize direction, scale by eps) is suboptimal. A natively L2-trained adversary might cause more damage at smaller eps, bringing the attack budget closer to certifiable radii. Constraint: adversary must not outperform clean PPO.

4. **Tighter certification methods**: GP is exact for the x^2+x architecture but the architecture itself limits radii. Explore certification methods for more expressive models (e.g., Lipschitz-bounded networks, interval bound propagation).

5. **Accept the gap for practical use**: The empirical detection works. The formal guarantee covers perturbations up to the certified radius; larger perturbations are detected with high probability but without formal guarantee.
