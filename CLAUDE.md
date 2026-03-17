# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Use `python3.8` (has all dependencies pre-installed). Run all scripts from the repo root.

```bash
pip install -r requirements.txt  # if setting up a new environment
```

Dependencies: `gymnasium`, `numpy`, `scipy`, `torch`, `stable-baselines3`.

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

### Architecture

Runtime-certified binary switcher for `CartPole-v1`. Chooses between PPO and LQR backup based on RS certification.

**Core decision rule:** Use PPO iff `pred == 0` (non-critical) AND `R >= delta_budget_l2`.

```
Raw obs (4D) → normalize → VanillaRSSwitcher.certify()
    → (pred, p_A_lower, R)  [R = sigma * Phi^{-1}(p_A_lower)]
    → CertifiedSwitcherController: allow_perf = (pred==0) and (R >= delta)
    → PPO or LQR
```

### Key files

| File | Role |
|------|------|
| `cartpole_ags_rs_switcher/models.py` | `SwitcherMLP`: 1-hidden-layer binary classifier |
| `cartpole_ags_rs_switcher/rs.py` | `VanillaRSSwitcher`: GPU-accelerated MC RS; `certify()` returns `(pred, p_A_lower, R_norm)` |
| `cartpole_ags_rs_switcher/evaluation.py` | Controller classes + `evaluate_controller()` |
| `cartpole_ags_rs_switcher/controllers.py` | `PerfPolicy` (PPO) and `QuantizedLQRBackup` |
| `cartpole_ags_rs_switcher/labeling.py` | `CriticalBurstLabeler`: PGD burst attacks to label critical/non-critical |
| `cartpole_ags_rs_switcher/attacks.py` | `pgd_l2_attack()`: targeted PGD in normalized obs space |
| `cartpole_ags_rs_switcher/training.py` | `train_switcher()`: BCE + noise augmentation |

### Conventions

- All L2 quantities (`epsilon_l2`, `delta_budget_l2`, `sigma`, `R`) in **normalized observation space**
- `p1 = P(critical)`: pred==0 means safe, pred==1 means critical
- Switcher checkpoint: `{"state_dict": ..., "obs_dim": int, "hidden_dim": int}`
- Dataset `.npz` keys: `X`, `y`, `state_mean`, `state_std`
- CartPole state_std ≈ `[0.045, 0.136, 0.0055, 0.202]` (pole_angle std 25× smaller than cart_vel)

### Key findings

- **Weak PPO (10k steps)**: clean return ~363, vulnerable to ε=0.5. Certified switcher: clean=500, attacked=500.
- **Strong PPO (20k+ steps)**: unbreakable — visits only equilibrium states, critical fraction = 0%.
- **Targeted PGD**: `attacks.py` uses `targeted=True` (default); minimizes `logit[clean] - logit[target]`.
- **Phase transition**: at ~19k steps PPO becomes fully robust; use 10k steps for meaningful experiments.

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

Hopper uses the **Zhang et al. optimal adversary** (`opt_attack`): a pre-trained `CtsPolicy(11→11)` network co-trained with PPO. Labeling is adversarial *detection* (not criticality):

- **y=0**: clean observation from PPO rollout
- **y=1**: `obs + tanh(attack_net(obs)) * eps` (eps=0.075, ZFilter-normalized)

The switcher detects adversarial perturbation. RS certifies the detection is robust to secondary perturbations.

### Key files

| File | Role |
|------|------|
| `hopper_ags_rs_switcher/controllers.py` | `HopperPerfPolicy` (PPO + Zhang attack) and `HopperBackupPolicy` (ATLA) |
| `hopper_ags_rs_switcher/evaluation.py` | `AlwaysPerfController`, `AlwaysBackupController`, `AnyTimeSwitcherController`, `evaluate_controller()` |
| `hopper_ags_rs_switcher/attacks.py` | `opt_attack()`: pre-trained adversary in normalized obs space |
| `hopper_ags_rs_switcher/labeling.py` | `HopperCriticalBurstLabeler`: detection dataset builder |
| `hopper_ags_rs_switcher/models.py` | Re-exports `SwitcherMLP` from CartPole module |
| `hopper_ags_rs_switcher/rs.py` | Re-exports `VanillaRSSwitcher` (GPU-accelerated) from CartPole module |
| `scripts/plot_hopper_results.py` | Generates figures from multi-seed JSON results |

### AnyTimeSwitcherController — 4-phase state machine

**Phase 1 — PPO monitoring**: Use PPO every step. RS-certify obs each step with `delta=0.075`. `detection_k=2` consecutive uncertified steps → attack declared → enter Phase 2. RS is a monitoring-only signal; per-step switching is not done (ZFilter churn).

**Phase 2 — ATLA recovery**: Use ATLA for `recovery_k=100` steps. Covers burst_k=75 steps + 25-step stabilization margin before PPO re-entry.

**Phase 3 — RS commit check**: RS-certify. First step with `pred==0` AND `R >= delta` → permanently commit to PPO. Forced commit after `commit_timeout_k=5` steps.

**Phase 4 — Committed PPO**: PPO for remainder of episode; no further RS calls. Justified by single-attack-per-episode threat model.

**Why not a loop back to Phase 1?** With `delta=0.075`, P(false alarm per step | clean) ≈ 10%. Re-entering Phase 1 continuously causes repeated false-alarm-triggered ATLA windows (≈90% time in ATLA). Permanent commit avoids this since the attack fires at most once per episode.

**False alarm probability**: P(false alarm per detection window) ≤ P(R < 0.075 | clean)^2 ≈ 0.01 for detection_k=2.

**L-inf → L-2 conversion**: Zhang attack is L-inf(ε=0.075). For obs_dim=11: L-2 budget = 0.075√11 ≈ 0.249. The empirical R_exec ≈ 0.30 > 0.249, confirming the commit gate is achievable.

### Benchmark results (3 seeds × 30 episodes)
*(burst_k=75, t_candidate_max=100, recovery_k=100, sigma=0.1, delta=0.075, n_samples=10000, GPU)*

| Controller | Clean return | Attacked return | Falls (clean) | Falls (attacked) |
|---|:---:|:---:|:---:|:---:|
| Always PPO | 3613 ± 134 | 1954 ± 1427 | 4/90 | **53/90** |
| Always ATLA | 2359 ± 954 | 2466 ± 1006 | 69/90 | 68/90 |
| **Anytime Switcher** | **3569 ± 300** | **3183 ± 882** | **5/90** | **22/90** |

Key observations:
- **PPO collapses** under 75-step early burst (return halved, 53/90 falls)
- **Switcher is robust**: return barely changes (3569 → 3183), falls 5 → 22/90
- **Clean cost is negligible**: Switcher ≈ PPO clean (3569 vs 3613, allow_perf=0.896)
- **ATLA alone unusable**: 69/90 falls even clean — cannot be standalone policy

### Checkpoint and path conventions

- **Attack checkpoint** (`Hopper_Attack_PPO.model`): contains `policy_model`, `adversary_policy_model`, and `envs[0]` (custom_env). All three MUST come from the same file for consistent ZFilter statistics.
- **ATLA checkpoint** (`Hopper_ATLA.model`): separate checkpoint; provides its own ZFilter via `custom_env.state_filter`.
- **custom_env API**: `reset(uState_12d, None, name="Hopper")` → normalized obs (11D); `step(action, change_filter=False, name="Hopper")` → `(result, norm_rew, is_done, info)` where `result[1]` is the next normalized obs.
- **old `policy_gradients/`**: renamed to `other_attacks/optimal_attack/opt_pg/` to avoid import conflict.

### Dual ZFilter normalization

At each step, TWO separate normalizations are in play:
1. **PPO ZFilter** (from `Hopper_Attack_PPO.model`'s `custom_env`): applied by `custom_env.step()` → `obs_ppo`
2. **ATLA ZFilter** (from `Hopper_ATLA.model`'s `custom_env.state_filter`): applied to raw sim obs → `obs_atla`

Raw 11D obs reconstructed via `_raw_obs_from_sim()`: `qpos[1:6] + clip(qvel[:6], -10, 10)`.

### gym 0.26 compatibility

`custom_env.env.step()` returns a 5-tuple in gym 0.26. `_patch_gym_env()` converts to 4-tuple. `custom_env.env.reset()` may return `(obs, info)`; `start_episode()` unpacks this.

### Episode termination

Custom Hopper termination (`height > 0.7 and |ang| < 0.2`) never fires for well-trained PPO. Hard `max_steps=1000` cap applied in `HopperPerfPolicy.step()`. A "fall" is any episode where `done=True` before step 1000.

---

## Future work / extension to other environments

The Hopper pipeline is designed to extend to HalfCheetah, Walker2d, and Ant:
1. Obtain `{Env}_PPO.model` + `{Env}_Attack_PPO.model` + `{Env}_ATLA.model` from Zhang et al.
2. `build_labels_{env}.py`: same detection labeling (clean/adv 50/50) — only `obs_dim` and `eps` change
3. `train_switcher_{env}.py`: identical to Hopper trainer
4. `evaluate_burst_attack_{env}.py`: same 3-controller setup; tune `burst_k`, `recovery_k`, `delta` for the env
5. `VanillaRSSwitcher` and `AnyTimeSwitcherController` are env-agnostic — no changes needed
