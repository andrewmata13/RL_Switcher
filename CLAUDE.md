# CLAUDE.md

## Setup

Use `python3.8` (has all dependencies pre-installed). Run all scripts from the repo root.

Dependencies: `gymnasium`, `numpy`, `scipy`, `torch`, `stable-baselines3`.

---

## Core Idea

At each step, choose between a **PPO policy** (high return, adversarially vulnerable) and a **safe backup** (robust, lower return). A binary **switcher** trained with Randomized Smoothing (RS) detects adversarial observations.

**Proper certification**: `certified_safe = (pred==0) AND (R >= delta)`. Never use pred-only — requires `R >= delta` for the formal guarantee.

**Labels**: y=0 (clean PPO obs), y=1 (opt_attacked obs). 50/50 split.

**Two certification backends**:
- **VanillaRSSwitcher** (MC RS): `SwitcherMLP`/`SwitcherDeepMLP`/`SwitcherRobustMLP`; statistical lower bound via Clopper-Pearson
- **GPSwitcher** (Gil-Pelaez): `SwitcherQuadMLP` with x²+x activation; exact analytical cert, ~2-5ms CPU, **preferred**

---

## Controllers

### ContinuousSwitcherController — **primary controller for paper**

Hysteresis-based loop (no permanent commit), handles multi-burst / repeated attacks:
```
PPO state: alarm_count += 1 if NOT certified; -= forgive_decay if certified
           alarm_count >= K_enter → ATLA
ATLA state: safe_count += 1 if certified; = 0 otherwise
            safe_count >= K_exit → PPO
```

**Key insight**: K_enter suppresses false alarms (P(K_enter consecutive alarms) exponentially small). forgive_decay=1.0 (don't tune this — it doesn't help).

**Attack modes**: `single` (one burst, random T), `multi` (n_bursts × burst_k, cooldown_k gap), `arbitrary` (Bernoulli per step).

### 4-Phase (single-burst, Hopper only)

```
Phase 1: PPO + RS-certify each step
  detection_k consecutive NOT-certified-safe → Phase 2
Phase 2: ATLA recovery
  Adaptive: recovery_confirm_k consecutive certified-safe → Phase 3
Phase 3: RS Commit Check (≤ commit_timeout_k steps)
  First certified-safe → Phase 4; forced commit after timeout
Phase 4: Committed PPO (permanent)
```

---

## Code Organization

### `rs_switcher_common/`

| File | Role |
|------|------|
| `env_config.py` | `EnvConfig` + `HOPPER`/`HALFCHEETAH`/`WALKER2D` + `ENV_REGISTRY` |
| `models.py` | `SwitcherMLP`, `SwitcherDeepMLP`, `SwitcherRobustMLP`, `load_switcher()` |
| `gp_models.py` | `SwitcherQuadMLP`, `GPSwitcher`, `load_gp_switcher()` |
| `rs.py` | `VanillaRSSwitcher`: certify() returns `(pred, p_A_lower, R)` |
| `controllers.py` | `MuJoCoPerfPolicy` (PPO + Zhang attack), `MuJoCoBackupPolicy` (ATLA), `raw_obs_from_sim()` |
| `evaluation.py` | `AlwaysPerfController`, `AlwaysBackupController`, `AdaptiveSwitcherController`, `ContinuousSwitcherController`, `evaluate_controller()` |
| `attacks.py` | `opt_attack()`: pre-trained adversary, L-inf and L2 |
| `labeling.py` | `CriticalBurstLabeler`, `collect_state_stats()` |
| `training.py` | `train_switcher()`: BCE + noise augmentation, AdamW + cosine LR |
| `clean_policies.py` | `CleanPerfPolicy`, `CleanBackupPolicy`, `PPOAsBackup` |
| `compat.py` | `patch_gym_env()` (gym 0.26 compatibility) |

### `cartpole_rs_switcher/`

CartPole-specific: PGD attack, LQR backup, certified switcher controller.

### Checkpoint conventions

- **Attack checkpoint** (`HalfCheetah_Attack_PPO.model`): contains `policy_model`, `adversary_policy_model`, `envs[0]`. All three MUST come from the same file.
- **ATLA checkpoint**: use `Attack_ATLA` file (not bare `ATLA.model`) for its own ZFilter.
- **`policy_gradients/`**: required at repo root for unpickling `.model` files. Do not delete.
- **Switcher ckpt keys**: `state_dict`, `obs_dim`, `model_type`, `hidden_dim(s)`. Use `load_switcher()` or `load_gp_switcher()`.
- **Dataset `.npz` keys**: `X`, `y`, `state_mean`, `state_std`
- All L2 quantities (`sigma`, `R`, `delta`, `eps`) are in **ZFilter-normalized obs space**

---

## GP Switcher Architecture

`SwitcherQuadMLP`: `Linear → x²+x → Linear(2)`. Exact analytical certification via Gil-Pelaez, no MC sampling.

**Architecture winner**: `SwitcherQuadMLP h=512` (~95% accuracy on HalfCheetah, ~3ms cert). Use quad h=512 by default.

**GP vs RS timing** (HalfCheetah, σ=0.2, same SwitcherQuadMLP model, `results/halfcheetah_same_model_h512_s02.json`):
RS needs ≥10k samples to approach GP's certified radius (0.284 vs 0.305), but RS@10k takes 13.4ms — over the 8ms real-time budget. GP certifies in 2.6ms with the highest radius.

**Sigma insight**: certify at `sigma_cert = sigma_train / 2` → higher p_A → larger R despite smaller sigma.

**`SwitcherRobustMLP`** (for MC RS): wide BN+Dropout network, trained with n_noise_copies=4 + cosine LR. Best accuracy (99%+).

---

## Per-Environment Results

### CartPole — toy example in paper (COMPLETE)

**Setup**: 15k-step PPO (`models/perf_cartpole_15000k.zip`), LQR backup (`QuantizedLQRBackup`), GP switcher. MuJoCo-style labeling (no criticality criterion — strong policies have no critical states). PGD L2 attack, eps=1.0, burst=200 oracle.

**Results** (50 episodes, median ± std):

| | Clean | Single (burst=200) | Multi (2×100) | Arbitrary (E=200) |
|---|---|---|---|---|
| Always PPO | 500 ± 0 | 107 ± 42 | 450 ± 98 | 500 ± 0 |
| Always LQR | 500 ± 0 | 500 ± 0 | 500 ± 0 | 500 ± 0 |
| GP Switcher (ours) | 500 ± 0 | **500 ± 0** | **500 ± 0** | **500 ± 0** |

Switcher % time in PPO: single=65.3%, multi=60.5%.

**Paper config**: `sigma=0.25`, `delta=0.25`, `delta_budget_l2=0.25`, `burst=200`, `eps=1.0`, `K_enter=5`, `K_exit=5`

**Artifacts**:
- PPO: `models/perf_cartpole_15000k.zip` (100% episodes reach 500 clean)
- Dataset: `data/cartpole_15k_dataset.npz` (100k samples, 50/50, MuJoCo-style labeling)
- GP switcher: `models/cartpole_15k_gp.pt` (sigma=0.25, hidden=64, 99.9% accuracy)
- Results: `results/cartpole_15k_gp2_clean.json`, `results/cartpole_15k_gp_bk200.json`, `results/cartpole_multi2x100.json`, `results/cartpole_arb200.json`

```bash
# Evaluate CartPole (attacked)
python3.8 scripts/evaluate_burst_attack.py \
    --perf-path models/perf_cartpole_15000k.zip \
    --gp-switcher-path models/cartpole_15k_gp.pt \
    --dataset data/cartpole_15k_dataset.npz \
    --sigma 0.25 --delta-budget-l2 0.25 \
    --episodes 50 --seed 0 --epsilon-l2 1.0 --burst-k 200 \
    --attack-mode oracle --output-json results/cartpole_result.json
```

**Note**: Do NOT use criticality-based labeling for CartPole — strong policies have no critical states. Build dataset by collecting clean PPO rollouts (y=0) and PGD-attacked versions (y=1) directly.

---

### HalfCheetah — **main env** (COMPLETE)

**Paper config** (continuous GP, multi-burst 3×100/100, L2 eps=0.5, 30 episodes):
- `sigma=0.2`, `delta=0.2`, `K_enter=5`, `K_exit=5`, `forgive_decay=1.0`

**Results** (30 episodes, median ± std):

| | Clean | Single (burst=300) | Multi (3×100) | Arbitrary (E=300) |
|---|---|---|---|---|
| Always PPO | 7245 ± 99 | 4082 ± 1290 | 3168 ± 1661 | 5929 ± 1210 |
| Always ATLA | 5652 ± 46 | 5641 ± 50 | 5649 ± 52 | 5632 ± 44 |
| Cont. GP (ours) | 7168 ± 125 | 6687 ± 135 | 6598 ± 1196 | 5874 ± 1934 |

Note: arbitrary attack result has high variance (±1934); the switcher barely beats always-PPO (5874 vs 5929 median). Multi-burst is the strongest result.

**Detection analysis** at delta=0.2: TNR=70.1%, TPR=79.7%, burst-level detection 100% for 100-step bursts (K_enter=5). Two-regime: ‖δ‖≤0.2 → provably correct; ‖δ‖>0.2 → certificate collapses → detection signal.

**Artifacts**:
- `models/halfcheetah_switcher_gp_s02.pt` (sigma=0.2, 95.2%)
- `data/halfcheetah_critical_dataset.npz`
- Results: `results/halfcheetah_paper_multi.json`, `results/halfcheetah_paper_single.json`, `results/halfcheetah_paper_arb300.json`
- R-distribution data: `results/halfcheetah_R_distribution.json`

```bash
python3.8 scripts/evaluate_continuous_controller.py --env halfcheetah \
    --perf-path HalfCheetah/HalfCheetah_PPO.model \
    --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --backup-path HalfCheetah/HalfCheetah_ATLA.model \
    --gp-switcher-path models/halfcheetah_switcher_gp_s02.pt \
    --dataset data/halfcheetah_critical_dataset.npz \
    --sigma 0.2 --delta-budget-l2 0.2 --episodes 30 --seed 0 \
    --attack-mode multi --n-bursts 3 --burst-k 100 --cooldown-k 100 \
    --K-enter 5 --K-exit 5 --forgive-decay 1.0 \
    --attack-norm l2 --attack-eps 0.5 \
    --output-json results/halfcheetah_paper_multi.json
```

---

### Hopper — secondary stories (NOT in paper)

**Story 1: Adaptive 4-phase + ATLA backup (single-burst)**
Clean pipeline (frozen normalization, no ZFilter). Use `--clean` flag.
- `sigma=0.1`, `delta=0.075`, `detection_k=10`, `recovery_confirm_k=10`, `commit_timeout_k=5`
- Clean 3480±172 (4% falls), Attacked 3250±702 (15% falls) — 100 eps

**Story 2: PPOAsBackup + continuous controller**
`PPOAsBackup` reads `raw_obs_from_sim()` (always clean). Same gait → 0% transition falls.
- `sigma_cert=0.05`, `delta=0.05`, `K_enter=2`, `K_exit=5`
- Single burst: 0/30 falls vs always_PPO 22/30

**Artifacts**: `Hopper/Hopper_Clean_PPO.pt`, `Hopper/Hopper_Clean_ATLA.pt`, `data/hopper_clean_ppoonly.npz`, `models/hopper_clean_gp.pt`

**Hopper v2** (in progress, not for paper): ATLA v2/v3 in `Hopper/atla_v2_ckpts/`. Gait mismatch unsolved.

**Critical past bug**: Do NOT store raw obs in dataset. Must store normalized obs. If state_mean[0] ≈ 1.3 (Hopper height), dataset has raw obs.

---

### Walker2D (INCOMPLETE, not in paper)

Continuous controller broken (returns 527-1178 vs always_backup 3544-3674). ATLA gait incompatibility.
Likely needs PPOAsBackup approach (not yet tried).

---

## Figures (poster_figures/)

Style file: `poster_figures/bak_matplotlib.mlpstyle` — all plot scripts should use this.

| Figure | Script | Data |
|--------|--------|------|
| `cert_collapse.pdf` | `plot_cert_collapse.py` | live rollout, seed=4; **primary figure** |
| `latency_wall.pdf` | `plot_latency_wall.py` | `results/halfcheetah_same_model_h512_s02.json` |
| `bar_chart.pdf` | `plot_bar_chart.py` | `results/halfcheetah_paper_*.json`, `results/cartpole_15k_gp*.json` |
| `r_distribution.pdf` | `plot_r_distribution.py` | `results/halfcheetah_R_distribution.json` |
| `episode_trace.pdf` | `plot_episode_trace.py` | live rollout, seed=4 (superseded by cert_collapse) |

**cert_collapse**: Signed certified radius (+R when pred=0 safe, −R when pred=1 attacked) on one HalfCheetah multi-burst episode. Shows certificate flipping to negative during attacks. Smoothed with 15-step rolling mean (~K_enter×3, one gait cycle).

**latency_wall**: GP vs RS scatter — certified radius (y) vs latency (x, log scale). Vertical line at 8ms real-time deadline. RS needs ≥10k samples to approach GP's radius, but at 13ms that fails real-time. GP: 2.6ms exact.

```bash
python3.8 scripts/plot_cert_collapse.py --seed 4
python3.8 scripts/plot_latency_wall.py
python3.8 scripts/plot_bar_chart.py
python3.8 scripts/plot_r_distribution.py
```

---

## MuJoCo Pipeline (build from scratch)

```bash
# 1. Build dataset (clean y=0, opt-attacked y=1)
python3.8 scripts/build_labels_mujoco.py --env halfcheetah \
    --perf-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
    --dataset-out data/halfcheetah_critical_dataset.npz \
    --episodes 20 --subsample-every 5

# 2. Train GP switcher
python3.8 scripts/train_switcher_gp.py \
    --dataset data/halfcheetah_critical_dataset.npz \
    --output models/halfcheetah_switcher_gp_s02.pt \
    --hidden-dim 512 --epochs 500 --sigma 0.2

# 3. Evaluate
python3.8 scripts/evaluate_continuous_controller.py --env halfcheetah \
    [see command above]
```

---

## Certification Gap Summary

| Env | sigma | Avg cert R | L2 eps that hurts PPO | Resolution |
|-----|-------|-----------|----------------------|------------|
| CartPole | 0.25 | ~0.4 | 1.0 | Two-regime detection |
| HalfCheetah | 0.2 | 0.323 | 0.50 | Two-regime detection |
| Hopper | 0.1 | 0.079 | 0.13 | small gap |
| Walker2D | 0.1 | 0.090 | 0.10 | small gap |

**Two-regime detection**: Within R → provably correct. Beyond R → certificate collapses → IS the detection signal. At delta=0.2 (HalfCheetah): 100% burst-level detection for 100-step bursts with K_enter=5.

---

## Paper Plan (ACNS 2026, 5 pages)

**Scope**: CartPole (toy) + HalfCheetah (main), continuous switcher only, multi-burst primary threat model.

**Structure**:

1. **Introduction** (~0.5p) — adversarial attacks on RL; gap: existing defenses are static or uncertified; contribution: runtime certified switching with repeated-attack guarantee.

2. **Background** (~0.5p) — Randomized Smoothing, Gil-Pelaez certification, threat model (L2 bounded, observation-only adversary, multi-burst).

3. **Method** (~1.5p)
   - GP Switcher: `SwitcherQuadMLP` (x²+x activation), training (BCE + noise aug, 50/50 clean/attacked), Gil-Pelaez exact cert
   - `ContinuousSwitcherController`: hysteresis loop (K_enter/K_exit), false-alarm bound (P(false alarm) ≤ p_A^K_enter), two-regime detection argument

4. **Experiments** (~1.5p)
   - CartPole: toy demonstration (PPO median 107 → Switcher 500 under burst=200, eps=1.0)
   - HalfCheetah: main table (clean / single burst=300 / multi 3×100 / arbitrary E=300), vs always-PPO and always-ATLA
   - Three figures: bar chart, R-distribution (certificate collapse), episode trace

5. **Conclusion** (~0.25p) — limitations (cert gap, Walker2D gait), future work.

**Figures** (all generated, in `poster_figures/`):
- `bar_chart.pdf` — main results (HalfCheetah + CartPole panels, broken y-axis)
- `r_distribution.pdf` — motivates two-regime detection argument
- `episode_trace.pdf` — qualitative: ATLA switch during attack window, seed=4

**Still needed**:
- Detection guarantee derivation / theorem box
- Related work (ATLA, SA-RL, Zhang et al., RS in RL)
