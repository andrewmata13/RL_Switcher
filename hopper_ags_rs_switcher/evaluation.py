"""
Controller classes and evaluation loop for Hopper.

Controllers:
  AlwaysPerfController   — always uses native PPO
  AlwaysBackupController — always uses ATLA backup
  AnyTimeSwitcherController — RS-certified detection + permanent PPO commit:
      Phase 1 (PPO monitoring): use PPO; RS-certify each step; detection_k
          consecutive failures → attack declared → Phase 2.
      Phase 2 (ATLA recovery): ATLA for recovery_k steps → Phase 3.
      Phase 3 (RS commit check): RS-certify; first certified step → Phase 4;
          forced commit after commit_timeout_k steps.
      Phase 4 (committed PPO): PPO for remainder of episode, no more RS calls.

The evaluation loop drives the simulation via HopperPerfPolicy.step() and
computes ATLA-normalized obs from the raw sim state at each step.
"""
from typing import Dict, List, Tuple
import numpy as np

from .controllers import HopperPerfPolicy, HopperBackupPolicy, _raw_obs_from_sim
from .rs import VanillaRSSwitcher
from .attacks import opt_attack


# ── Controller classes ────────────────────────────────────────────────────────

class AlwaysPerfController:
    def __init__(self, perf: HopperPerfPolicy):
        self.perf = perf

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:
        action = self.perf.predict(obs_ppo)
        return action, {
            "allow_perf": 1.0, "p_critical": 0.0,
            "R_rs": float("nan"), "R_exec": float("nan"),
        }


class AlwaysBackupController:
    def __init__(self, backup: HopperBackupPolicy):
        self.backup = backup

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:
        action = self.backup.predict(obs_atla)
        return action, {
            "allow_perf": 0.0, "p_critical": 1.0,
            "R_rs": float("nan"), "R_exec": float("nan"),
        }


class AnyTimeSwitcherController:
    """
    RS-certified any-time attack detection with permanent PPO commit.

    Phase 1 — PPO monitoring (RS detection):
        Use PPO.  RS-certify obs_ppo each step.  A step is "not certified safe"
        if pred == 1 OR R < delta_budget_l2.  detection_k consecutive unsafe
        steps → attack declared, enter Phase 2.

        P(false alarm per window) <= P(R < delta | clean)^detection_k.
        With delta=0.075 and sigma=0.1, P(R >= delta | clean) ~ 0.90, so
        P(false alarm) <= 0.10^2 = 0.01 for detection_k=2.

    Phase 2 — ATLA recovery (recovery_k steps):
        Use ATLA regardless of obs.  recovery_k >= burst_k covers the full burst;
        extra steps allow Hopper to restabilize before PPO re-entry.

    Phase 3 — RS commit check:
        RS-certify obs_ppo.  First step with pred==0 AND R >= delta_budget_l2
        → permanently commit to PPO (Phase 4).  Forced commit after
        commit_timeout_k steps without certification.

    Phase 4 — Committed PPO:
        Use PPO for the remainder of the episode; no further RS calls.
        Justified by single-attack-per-episode threat model (attack fires at
        most once per episode, so monitoring after recovery is unnecessary).

    GPU acceleration: pass a VanillaRSSwitcher constructed with device="cuda".
    """

    _PPO       = "ppo"
    _ATLA      = "atla"
    _RS_CHECK  = "rs_check"
    _COMMITTED = "committed"

    def __init__(self, perf: HopperPerfPolicy, backup: HopperBackupPolicy,
                 rs: VanillaRSSwitcher,
                 delta_budget_l2: float,
                 detection_k: int = 2,
                 recovery_k: int = 100,
                 commit_timeout_k: int = 5):
        self.perf              = perf
        self.backup            = backup
        self.rs                = rs
        self.delta_budget_l2   = delta_budget_l2
        self.detection_k       = detection_k
        self.recovery_k        = recovery_k
        self.commit_timeout_k  = commit_timeout_k
        self._reset()

    def _reset(self):
        self._phase              = self._PPO
        self._consec_unsafe      = 0
        self._recovery_remaining = 0
        self._commit_steps       = 0

    def reset_episode(self) -> None:
        self._reset()

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:

        # ── Phase 4: committed PPO (no more RS calls) ──────────────────────────
        if self._phase == self._COMMITTED:
            return self.perf.predict(obs_ppo), {
                "allow_perf": 1.0, "p_critical": 0.0,
                "R_rs": float("nan"), "R_exec": float("nan"),
            }

        # ── Phase 1: certified PPO monitoring ─────────────────────────────────
        if self._phase == self._PPO:
            pred, p_A_lower, R = self.rs.certify(obs_ppo)
            certified_safe = (pred == 0) and (R >= self.delta_budget_l2)

            if not certified_safe:
                self._consec_unsafe += 1
            else:
                self._consec_unsafe = 0

            if self._consec_unsafe >= self.detection_k:
                self._phase              = self._ATLA
                self._recovery_remaining = self.recovery_k
                self._consec_unsafe      = 0
                p_crit = (1.0 - p_A_lower) if pred == 0 else p_A_lower
                return self.backup.predict(obs_atla), {
                    "allow_perf": 0.0, "p_critical": p_crit,
                    "R_rs": R, "R_exec": float("nan"),
                }
            else:
                # Phase 1 always uses PPO — RS cert is a detection monitor only,
                # not a per-step gate. Switching on individual uncertified steps
                # causes ZFilter churn (step-level PPO↔ATLA incompatibility).
                return self.perf.predict(obs_ppo), {
                    "allow_perf": 1.0,
                    "p_critical": 1.0 - p_A_lower if pred == 0 else p_A_lower,
                    "R_rs": R, "R_exec": float("nan"),
                }

        # ── Phase 2: ATLA recovery window ─────────────────────────────────────
        if self._phase == self._ATLA:
            self._recovery_remaining -= 1
            if self._recovery_remaining <= 0:
                self._phase        = self._RS_CHECK
                self._commit_steps = 0
            return self.backup.predict(obs_atla), {
                "allow_perf": 0.0, "p_critical": 1.0,
                "R_rs": float("nan"), "R_exec": float("nan"),
            }

        # ── Phase 3: RS commit check ───────────────────────────────────────────
        # Forced commit after timeout → permanently commit to PPO
        if self._commit_steps >= self.commit_timeout_k:
            self._phase = self._COMMITTED
            return self.perf.predict(obs_ppo), {
                "allow_perf": 1.0, "p_critical": 0.0,
                "R_rs": float("nan"), "R_exec": float("nan"),
            }

        pred, p_A_lower, R = self.rs.certify(obs_ppo)
        self._commit_steps += 1

        committed = (pred == 0 and R >= self.delta_budget_l2)
        if committed:
            self._phase = self._COMMITTED
            action = self.perf.predict(obs_ppo)
        else:
            action = self.backup.predict(obs_atla)

        p_crit = (1.0 - p_A_lower) if pred == 0 else p_A_lower
        return action, {
            "allow_perf": float(committed),
            "p_critical": p_crit,
            "R_rs": R, "R_exec": R,
        }


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_controller(
    controller,
    perf: HopperPerfPolicy,
    backup: HopperBackupPolicy,
    n_episodes: int = 10,
    seed: int = 0,
    attack: bool = False,
    burst_k: int = 20,
    horizon: int = 1000,
    t_candidate_max: int = 20,
) -> Tuple[List[float], List[Dict]]:
    """
    Run n_episodes of Hopper with the given controller.

    PPO's custom_env drives the simulation.  At each step, ATLA-normalized obs
    is computed from the raw sim state using ATLA's own read-only ZFilter.

    Burst attack model
    ------------------
    When attack=True, a single K-step adversarial burst is injected once per
    episode.  The burst fires exactly at step T_candidate, sampled uniformly
    from [0, t_candidate_max].

    t_candidate_max=20 (default): attack fires in the early transient phase;
    the committed switcher's fixed warmup covers this window.

    t_candidate_max=900 (any-time): attack can fire anywhere in the episode,
    exposing controllers that rely on a fixed warmup window.

    The burst lasts exactly burst_k steps, after which the episode continues
    normally.  PPO obs are perturbed; ATLA obs (from raw sim state) are clean.
    """
    np.random.seed(seed)
    returns: List[float] = []
    logs:    List[Dict]  = []

    for ep in range(n_episodes):
        obs_ppo = perf.start_episode()
        done    = False
        t       = 0
        step_logs: List[Dict] = []

        # Reset episode-level state for controllers that track it
        if hasattr(controller, "reset_episode"):
            controller.reset_episode()

        # Per-episode burst: fires at a random step in [0, t_candidate_max]
        T_candidate    = np.random.randint(0, t_candidate_max + 1)
        burst_remaining = 0

        while not done and t < horizon:
            # ATLA obs from same raw sim state
            raw      = _raw_obs_from_sim(perf.custom_env)
            obs_atla = backup.normalize(raw)

            # ── Burst attack logic ──────────────────────────────────────────
            obs_ctrl = obs_ppo
            if attack and t == T_candidate:
                burst_remaining = burst_k

            if attack and burst_remaining > 0 and perf.attack_model is not None:
                obs_ctrl = opt_attack(perf.attack_model, obs_ppo, eps=perf.eps)
                burst_remaining -= 1
            # ───────────────────────────────────────────────────────────────

            action, info = controller.select(obs_ctrl, obs_atla)
            obs_ppo, _, done, _ = perf.step(action)
            step_logs.append(info)
            t += 1

        ep_return = perf.custom_env.total_true_reward
        returns.append(ep_return)

        # A "fall" is any episode that terminated before the horizon cap
        fell = done and (t < horizon)

        # Aggregate per-episode metrics
        allow_mean = float(np.mean([l["allow_perf"] for l in step_logs]))
        R_vals     = [l["R_exec"] for l in step_logs if not np.isnan(l["R_exec"])]
        R_mean     = float(np.nanmean(R_vals)) if R_vals else float("nan")
        logs.append({"allow_perf": allow_mean, "R_exec": R_mean, "fell": fell, "ep_len": t})

    return returns, logs
