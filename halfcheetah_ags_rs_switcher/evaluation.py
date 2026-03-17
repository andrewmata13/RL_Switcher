"""
Controller classes and evaluation loop for HalfCheetah.

Controllers:
  AlwaysPerfController      — always uses native PPO
  AlwaysBackupController    — always uses ATLA backup
  AnyTimeSwitcherController — RS-certified detection + permanent PPO commit:
      Phase 1 (PPO monitoring): RS-certify each step; detection_k consecutive
          failures → attack declared → Phase 2.
      Phase 2 (ATLA recovery): ATLA for recovery_k steps → Phase 3.
      Phase 3 (RS commit check): first certified step → Phase 4;
          forced commit after commit_timeout_k steps.
      Phase 4 (committed PPO): PPO for remainder of episode, no RS calls.

Mirrors hopper_ags_rs_switcher/evaluation.py exactly; only env-specific
calls (name="Cheetah", _raw_obs_from_sim) differ.
"""
from typing import Dict, List, Tuple
import numpy as np

from .controllers import CheetahPerfPolicy, CheetahBackupPolicy, _raw_obs_from_sim
from .rs import VanillaRSSwitcher
from .attacks import opt_attack


# ── Controller classes ────────────────────────────────────────────────────────

class AlwaysPerfController:
    def __init__(self, perf: CheetahPerfPolicy):
        self.perf = perf

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:
        return self.perf.predict(obs_ppo), {
            "allow_perf": 1.0, "p_critical": 0.0,
            "R_rs": float("nan"), "R_exec": float("nan"),
        }


class AlwaysBackupController:
    def __init__(self, backup: CheetahBackupPolicy):
        self.backup = backup

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:
        return self.backup.predict(obs_atla), {
            "allow_perf": 0.0, "p_critical": 1.0,
            "R_rs": float("nan"), "R_exec": float("nan"),
        }


class AnyTimeSwitcherController:
    """
    RS-certified any-time attack detection with permanent PPO commit.
    Identical logic to Hopper's AnyTimeSwitcherController.
    """

    _PPO       = "ppo"
    _ATLA      = "atla"
    _RS_CHECK  = "rs_check"
    _COMMITTED = "committed"

    def __init__(self, perf: CheetahPerfPolicy, backup: CheetahBackupPolicy,
                 rs: VanillaRSSwitcher,
                 delta_budget_l2: float,
                 detection_k: int = 2,
                 recovery_k: int = 100,
                 commit_timeout_k: int = 5):
        self.perf             = perf
        self.backup           = backup
        self.rs               = rs
        self.delta_budget_l2  = delta_budget_l2
        self.detection_k      = detection_k
        self.recovery_k       = recovery_k
        self.commit_timeout_k = commit_timeout_k
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

        # ── Phase 4: committed PPO ─────────────────────────────────────────────
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
    perf: CheetahPerfPolicy,
    backup: CheetahBackupPolicy,
    n_episodes: int = 10,
    seed: int = 0,
    attack: bool = False,
    burst_k: int = 20,
    horizon: int = 1000,
    t_candidate_max: int = 100,
) -> Tuple[List[float], List[Dict]]:
    """
    Run n_episodes of HalfCheetah with the given controller.

    Burst attack model: when attack=True, a single burst_k-step adversarial
    burst fires at step T ~ U[0, t_candidate_max] once per episode.
    PPO obs are perturbed; ATLA obs (from raw sim state) are clean.
    """
    np.random.seed(seed)
    returns: List[float] = []
    logs:    List[Dict]  = []

    for ep in range(n_episodes):
        obs_ppo = perf.start_episode()
        done    = False
        t       = 0
        step_logs: List[Dict] = []

        if hasattr(controller, "reset_episode"):
            controller.reset_episode()

        T_candidate     = np.random.randint(0, t_candidate_max + 1)
        burst_remaining = 0

        while not done and t < horizon:
            raw      = _raw_obs_from_sim(perf.custom_env)
            obs_atla = backup.normalize(raw)

            obs_ctrl = obs_ppo
            if attack and t == T_candidate:
                burst_remaining = burst_k

            if attack and burst_remaining > 0 and perf.attack_model is not None:
                obs_ctrl = opt_attack(perf.attack_model, obs_ppo, eps=perf.eps)
                burst_remaining -= 1

            action, info = controller.select(obs_ctrl, obs_atla)
            obs_ppo, _, done, _ = perf.step(action)
            step_logs.append(info)
            t += 1

        ep_return = perf.custom_env.total_true_reward
        returns.append(ep_return)

        fell = done and (t < horizon)

        allow_mean = float(np.mean([l["allow_perf"] for l in step_logs]))
        R_vals     = [l["R_exec"] for l in step_logs if not np.isnan(l["R_exec"])]
        R_mean     = float(np.nanmean(R_vals)) if R_vals else float("nan")
        logs.append({"allow_perf": allow_mean, "R_exec": R_mean, "fell": fell, "ep_len": t})

    return returns, logs
