"""
Collect per-step certified radius R distributions for clean vs attacked observations.
Outputs data for histogram plots and detection ROC curves.

Usage:
    python3.8 scripts/collect_R_distribution.py --env halfcheetah \
        --perf-path HalfCheetah/HalfCheetah_PPO.model \
        --attack-path HalfCheetah/HalfCheetah_Attack_PPO.model \
        --backup-path HalfCheetah/HalfCheetah_ATLA.model \
        --gp-switcher-path models/halfcheetah_switcher_gp_s02.pt \
        --dataset data/halfcheetah_critical_dataset.npz \
        --sigma 0.2 --episodes 10 --seed 0 \
        --attack-mode multi --n-bursts 3 --burst-k 100 --cooldown-k 100 \
        --attack-norm l2 --attack-eps 0.5 \
        --output-json results/halfcheetah_R_distribution.json
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json
import numpy as np
import torch

from rs_switcher_common.env_config import ENV_REGISTRY
from rs_switcher_common.controllers import MuJoCoPerfPolicy, MuJoCoBackupPolicy
from rs_switcher_common.evaluation import (
    ContinuousSwitcherController, evaluate_controller,
)
from rs_switcher_common.gp_models import load_gp_switcher, GPSwitcher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="halfcheetah")
    p.add_argument("--perf-path", required=True)
    p.add_argument("--attack-path", required=True)
    p.add_argument("--backup-path", required=True)
    p.add_argument("--gp-switcher-path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--delta-budget-l2", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--attack-mode", default="multi")
    p.add_argument("--burst-k", type=int, default=100)
    p.add_argument("--n-bursts", type=int, default=3)
    p.add_argument("--cooldown-k", type=int, default=100)
    p.add_argument("--t-candidate-max", type=int, default=100)
    p.add_argument("--attack-norm", default="l2")
    p.add_argument("--attack-eps", type=float, default=0.5)

    p.add_argument("--K-enter", type=int, default=5)
    p.add_argument("--K-exit", type=int, default=5)
    p.add_argument("--forgive-decay", type=float, default=1.0)

    p.add_argument("--output-json", default="results/R_distribution.json")
    args = p.parse_args()

    config = ENV_REGISTRY[args.env]
    perf = MuJoCoPerfPolicy.load(config, args.perf_path, attack_path=args.attack_path)
    backup = MuJoCoBackupPolicy.load(config, args.backup_path)

    data = np.load(args.dataset)
    mean, std = data["state_mean"], data["state_std"]
    gp_ck = torch.load(args.gp_switcher_path, map_location="cpu")
    gp_model = load_gp_switcher(gp_ck)
    cert = GPSwitcher(gp_model, mean, std, sigma=args.sigma, device="cpu")

    ctrl = ContinuousSwitcherController(
        perf, backup, cert,
        delta_budget_l2=args.delta_budget_l2,
        K_enter=args.K_enter, K_exit=args.K_exit,
        forgive_decay=args.forgive_decay,
    )

    # Run attacked episodes (collects both clean and attacked step R values)
    print(f"Running {args.episodes} attacked episodes...")
    returns, logs = evaluate_controller(
        ctrl, perf, backup,
        n_episodes=args.episodes, seed=args.seed,
        attack=True, burst_k=args.burst_k,
        t_candidate_max=args.t_candidate_max,
        attack_norm=args.attack_norm, attack_eps=args.attack_eps,
        attack_mode=args.attack_mode,
        n_bursts=args.n_bursts, cooldown_k=args.cooldown_k,
    )

    # Collect all step data split by clean/attacked
    clean_steps = []   # (pred, R) tuples
    attacked_steps = []
    for ep_log in logs:
        for step in ep_log.get("R_per_step", []):
            entry = (step.get("pred", -1), step["R"])
            if step["attacked"]:
                attacked_steps.append(entry)
            else:
                clean_steps.append(entry)

    clean_preds = np.array([s[0] for s in clean_steps])
    clean_R = np.array([s[1] for s in clean_steps])
    atk_preds = np.array([s[0] for s in attacked_steps])
    atk_R = np.array([s[1] for s in attacked_steps])

    print(f"\n=== Raw prediction accuracy ===")
    print(f"  Clean steps ({len(clean_preds)}):  pred=0 (correct): {np.mean(clean_preds==0):.1%}  "
          f"pred=1 (false alarm): {np.mean(clean_preds==1):.1%}")
    print(f"  Attacked steps ({len(atk_preds)}): pred=1 (correct): {np.mean(atk_preds==1):.1%}  "
          f"pred=0 (missed): {np.mean(atk_preds==0):.1%}")

    print(f"\n=== R distribution (for pred=0 class, i.e. 'non-adversarial' prediction) ===")
    clean_pred0_R = clean_R[clean_preds == 0]
    atk_pred0_R = atk_R[atk_preds == 0]
    if len(clean_pred0_R) > 0:
        print(f"  Clean, pred=0 ({len(clean_pred0_R)} steps):  "
              f"mean R={np.mean(clean_pred0_R):.4f}  median={np.median(clean_pred0_R):.4f}")
    if len(atk_pred0_R) > 0:
        print(f"  Attacked, pred=0 ({len(atk_pred0_R)} steps): "
              f"mean R={np.mean(atk_pred0_R):.4f}  median={np.median(atk_pred0_R):.4f}")
    else:
        print(f"  Attacked, pred=0: 0 steps (all attacked obs classified as adversarial)")

    # Detection = NOT certified_safe = (pred != 0) OR (pred==0 AND R < delta)
    # certified_safe = (pred == 0) AND (R >= delta)
    deltas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    print(f"\n{'delta':>6s}  {'Clean certified':>15s}  {'Attacked detected':>17s}  "
          f"{'FPR':>6s}  {'FNR':>6s}")
    detection_metrics = []
    for d in deltas:
        clean_cert = (clean_preds == 0) & (clean_R >= d)
        atk_detected = ~((atk_preds == 0) & (atk_R >= d))
        tnr = float(np.mean(clean_cert))       # clean steps that pass certification
        tpr = float(np.mean(atk_detected))      # attacked steps that fail certification
        fpr = 1.0 - tnr                         # false alarm rate
        fnr = 1.0 - tpr                         # missed detection rate
        print(f"{d:6.2f}  {tnr:15.3f}  {tpr:17.3f}  {fpr:6.3f}  {fnr:6.3f}")
        detection_metrics.append({"delta": d, "TNR": tnr, "TPR": tpr, "FPR": fpr, "FNR": fnr})

    # Certified false alarm bound for K_enter consecutive steps
    print(f"\nCertified false alarm bounds (K_enter={args.K_enter}):")
    for d in [0.1, 0.2, 0.3]:
        clean_cert = (clean_preds == 0) & (clean_R >= d)
        fpr = 1.0 - float(np.mean(clean_cert))
        p_false_entry = fpr ** args.K_enter
        print(f"  delta={d:.1f}: P(false alarm/step)={fpr:.4f}, "
              f"P({args.K_enter} consecutive)={p_false_entry:.2e}")

    # Save
    out = {
        "clean_preds": [int(p) for p in clean_preds],
        "clean_R": [float(r) for r in clean_R],
        "attacked_preds": [int(p) for p in atk_preds],
        "attacked_R": [float(r) for r in atk_R],
        "detection_metrics": detection_metrics,
        "config": {
            "env": args.env, "sigma": args.sigma,
            "delta_budget_l2": args.delta_budget_l2,
            "attack_mode": args.attack_mode, "burst_k": args.burst_k,
            "n_bursts": args.n_bursts, "cooldown_k": args.cooldown_k,
            "attack_norm": args.attack_norm, "attack_eps": args.attack_eps,
            "K_enter": args.K_enter, "K_exit": args.K_exit,
            "episodes": args.episodes, "seed": args.seed,
        },
        "summary": {
            "n_clean": len(clean_R), "n_attacked": len(atk_R),
            "clean_pred0_rate": float(np.mean(clean_preds == 0)),
            "attacked_pred1_rate": float(np.mean(atk_preds == 1)),
            "R_clean_pred0_mean": float(np.mean(clean_pred0_R)) if len(clean_pred0_R) > 0 else None,
            "R_attacked_pred0_mean": float(np.mean(atk_pred0_R)) if len(atk_pred0_R) > 0 else None,
        },
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
