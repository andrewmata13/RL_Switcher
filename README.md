# CartPole AGS + RS Certified Switcher

A clean prototype for a **runtime-certified binary switcher** on `CartPole-v1`.

The project trains or loads:
- a **performance PPO policy**
- a **quantized LQR backup controller**
- a **binary switcher** that predicts whether a short adversarial observation burst would critically damage the performance policy
- a **regionized AGS surrogate** for the switcher
- an **exact binary randomized smoothing certificate** for the surrogate switcher

## Project structure

```text
cartpole_ags_rs_switcher/
  README.md
  requirements.txt
  cartpole_ags_rs_switcher/
    __init__.py
    config.py
    utils.py
    controllers.py
    models.py
    ags.py
    attacks.py
    labeling.py
    training.py
    evaluation.py
  scripts/
    train_perf.py
    build_labels.py
    train_switcher.py
    evaluate_switcher.py
```

## Install

```bash
pip install -r requirements.txt
```

## Typical workflow

### 1. Train the performance PPO policy

```bash
python scripts/train_perf.py --output models/perf_cartpole_ppo.zip
```

### 2. Build critical-state labels

```bash
python scripts/build_labels.py \
  --perf-path models/perf_cartpole_ppo.zip \
  --dataset-out data/critical_dataset.npz
```

### 3. Train the binary switcher

```bash
python scripts/train_switcher.py \
  --dataset data/critical_dataset.npz \
  --output models/switcher.pt
```

### 4. Evaluate certified switching

```bash
python scripts/evaluate_switcher.py \
  --perf-path models/perf_cartpole_ppo.zip \
  --switcher-path models/switcher.pt \
  --dataset data/critical_dataset.npz
```

## Notes

- This is a **prototype**, not the final paper implementation.
- The backup controller is **quantized LQR** for standard discrete `CartPole-v1`.
- The critical attack is a **short FGSM-style observation burst** against the PPO policy.
- The AGS surrogate is a **practical regionized diagonal moment-matching surrogate** for a 1-hidden-layer binary switcher.
- The binary RS certificate is exact for the deployed surrogate switcher:

\[
R(y)=\frac{\sigma}{2}\left(\Phi^{-1}(p_A)-\Phi^{-1}(p_B)\right).
\]

## Main runtime rule

The controller executes the performance policy only when the smoothed permission decision is positive and the certified radius exceeds the uncertainty budget:

\[
\text{use perf} \iff \bar g(y)=1 \;\wedge\; R_{\text{exec}}(y) \ge \Delta,
\]

where

\[
R_{\text{exec}}(y)=\min\{R_{\text{RS}}(y), R_{\text{reg}}(y)\}.
\]
