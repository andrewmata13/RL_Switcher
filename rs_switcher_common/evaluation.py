"""
Controller classes and evaluation loop for MuJoCo environments.

Controllers:
  AlwaysPerfController   — always uses native PPO
  AlwaysBackupController — always uses ATLA backup
  AnyTimeSwitcherController — RS-certified detection + permanent PPO commit:
      Phase 1 (PPO monitoring): use PPO; RS-certify each step; detection_k
          consecutive failures -> attack declared -> Phase 2.
      Phase 2 (ATLA recovery): ATLA for recovery_k steps -> Phase 3.
      Phase 3 (RS commit check): RS-certify; first certified step -> Phase 4;
          forced commit after commit_timeout_k steps.
      Phase 4 (committed PPO): PPO for remainder of episode, no more RS calls.
  AdaptiveSwitcherController — like AnyTimeSwitcher but Phase 2 actively
      monitors via RS instead of waiting a fixed recovery_k window.  The
      controller exits ATLA as soon as recovery_confirm_k consecutive
      certified-safe steps are observed, making it robust to varying attack
      lengths without knowing burst_k in advance.
  ContinuousSwitcherController — hysteresis-based switching for arbitrary
      attack models.  No permanent commit; loops between PPO and ATLA
      indefinitely.  Uses asymmetric thresholds (K_enter, K_exit) to
      control the false alarm / missed detection tradeoff.

The evaluation loop drives the simulation via MuJoCoPerfPolicy.step() and
computes ATLA-normalized obs from the raw sim state at each step.
"""
from typing import Dict, List, Tuple
import collections
import numpy as np

from .controllers import MuJoCoPerfPolicy, MuJoCoBackupPolicy, raw_obs_from_sim
from .rs import VanillaRSSwitcher
from .attacks import opt_attack


# -- Controller classes --------------------------------------------------------

class AlwaysPerfController:
    def __init__(self, perf: MuJoCoPerfPolicy):
        self.perf = perf

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:
        return self.perf.predict(obs_ppo), {
            "allow_perf": 1.0, "p_critical": 0.0,
            "R_rs": float("nan"), "R_exec": float("nan"),
        }


class AlwaysBackupController:
    def __init__(self, backup: MuJoCoBackupPolicy):
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

    Phase 1 -- PPO monitoring (RS detection):
        Use PPO.  RS-certify obs_ppo each step.  A step is "not certified safe"
        if pred == 1 OR R < monitoring_delta.  detection_k consecutive unsafe
        steps -> attack declared, enter Phase 2.

    Phase 2 -- ATLA recovery (recovery_k steps):
        Use ATLA regardless of obs.  recovery_k >= burst_k covers the full burst;
        extra steps allow the agent to restabilize before PPO re-entry.

    Phase 3 -- RS commit check:
        RS-certify obs_ppo.  First step with pred==0 AND R >= delta_budget_l2
        -> permanently commit to PPO (Phase 4).  Forced commit after
        commit_timeout_k steps without certification.

    Phase 4 -- Committed PPO:
        Use PPO for the remainder of the episode; no further RS calls.
        Justified by single-attack-per-episode threat model.

    GPU acceleration: pass a VanillaRSSwitcher constructed with device="cuda".
    """

    _PPO = "ppo"
    _ATLA = "atla"
    _RS_CHECK = "rs_check"
    _COMMITTED = "committed"

    def __init__(self, perf: MuJoCoPerfPolicy, backup: MuJoCoBackupPolicy,
                 rs: VanillaRSSwitcher,
                 delta_budget_l2: float,
                 detection_k: int = 2,
                 recovery_k: int = 100,
                 commit_timeout_k: int = 5,
                 monitoring_delta: float = None,
                 commit_steps: int = None):
        self.perf = perf
        self.backup = backup
        self.rs = rs
        self.delta_budget_l2 = delta_budget_l2
        # monitoring_delta: threshold used in Phase 1 detection.
        # If None, uses delta_budget_l2 (default behavior).
        # Set to 0.0 to use pred-only detection (ignoring R in Phase 1).
        self.monitoring_delta = delta_budget_l2 if monitoring_delta is None else monitoring_delta
        self.detection_k = detection_k
        self.recovery_k = recovery_k
        self.commit_timeout_k = commit_timeout_k
        # commit_steps: if set, Phase 4 lasts this many steps then loops back
        # to Phase 1 monitoring. If None, Phase 4 is permanent (single-burst).
        self.commit_steps = commit_steps
        self._reset()

    def _reset(self):
        self._phase = self._PPO
        self._consec_unsafe = 0
        self._recovery_remaining = 0
        self._commit_steps = 0
        self._commit_timer = 0

    def reset_episode(self) -> None:
        self._reset()

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:

        # -- Phase 4: committed PPO (no more RS calls) ------------------------
        if self._phase == self._COMMITTED:
            if self.commit_steps is not None:
                self._commit_timer += 1
                if self._commit_timer >= self.commit_steps:
                    # Finite commit: loop back to Phase 1 monitoring
                    self._phase = self._PPO
                    self._consec_unsafe = 0
                    self._commit_timer = 0
            return self.perf.predict(obs_ppo), {
                "allow_perf": 1.0, "p_critical": 0.0,
                "R_rs": float("nan"), "R_exec": float("nan"),
            }

        # -- Phase 1: certified PPO monitoring --------------------------------
        if self._phase == self._PPO:
            pred, p_A_lower, R = self.rs.certify(obs_ppo)
            certified_safe = (pred == 0) and (R >= self.monitoring_delta)

            if not certified_safe:
                self._consec_unsafe += 1
            else:
                self._consec_unsafe = 0

            if self._consec_unsafe >= self.detection_k:
                self._phase = self._ATLA
                self._recovery_remaining = self.recovery_k
                self._consec_unsafe = 0
                p_crit = (1.0 - p_A_lower) if pred == 0 else p_A_lower
                return self.backup.predict(obs_atla), {
                    "allow_perf": 0.0, "p_critical": p_crit,
                    "R_rs": R, "R_exec": float("nan"),
                }
            else:
                # Phase 1 always uses PPO -- RS cert is a detection monitor
                # only, not a per-step gate.  Switching on individual
                # uncertified steps causes ZFilter churn.
                return self.perf.predict(obs_ppo), {
                    "allow_perf": 1.0,
                    "p_critical": 1.0 - p_A_lower if pred == 0 else p_A_lower,
                    "R_rs": R, "R_exec": float("nan"),
                }

        # -- Phase 2: ATLA recovery window ------------------------------------
        if self._phase == self._ATLA:
            self._recovery_remaining -= 1
            if self._recovery_remaining <= 0:
                self._phase = self._RS_CHECK
                self._commit_steps = 0
            return self.backup.predict(obs_atla), {
                "allow_perf": 0.0, "p_critical": 1.0,
                "R_rs": float("nan"), "R_exec": float("nan"),
            }

        # -- Phase 3: RS commit check -----------------------------------------
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


class AdaptiveSwitcherController:
    """
    RS-certified any-time attack detection with adaptive backup monitoring.

    Identical to AnyTimeSwitcherController except Phase 2 actively RS-certifies
    while running ATLA, exiting as soon as recovery_confirm_k consecutive
    certified-safe steps are observed.  This makes the controller robust to
    varying attack lengths without requiring prior knowledge of burst_k.

    Phase 1 -- PPO monitoring (RS detection):
        Same as AnyTimeSwitcherController.  detection_k consecutive unsafe
        steps -> attack declared -> Phase 2.

    Phase 2 -- ATLA adaptive recovery:
        Use ATLA.  RS-certify obs_ppo each step.
        A step is "certified safe" if pred==0 AND R >= delta_budget_l2.
        recovery_confirm_k consecutive certified-safe steps -> Phase 3.
        No hard cap: during an active attack the adversarial obs will
        consistently fail certification, so the controller naturally stays in
        ATLA for the full attack duration without needing a timer.

    Phase 3 -- RS commit check:
        Same as AnyTimeSwitcherController.  First certified step -> Phase 4;
        forced commit after commit_timeout_k steps.

    Phase 4 -- Committed PPO:
        PPO for the remainder of the episode; no further RS calls.
        Justified by single-attack-per-episode threat model.

    Key difference vs AnyTimeSwitcherController:
        recovery_k (fixed window, must be tuned to burst_k) ->
        recovery_confirm_k (adaptive: count consecutive clean steps after
        attack ends).  recovery_confirm_k is independent of attack length;
        it measures environment stabilisation time, not attack duration.
    """

    _PPO = "ppo"
    _ATLA = "atla"
    _RS_CHECK = "rs_check"
    _COMMITTED = "committed"

    def __init__(self, perf: MuJoCoPerfPolicy, backup: MuJoCoBackupPolicy,
                 rs: VanillaRSSwitcher,
                 delta_budget_l2: float,
                 detection_k: int = 2,
                 recovery_confirm_k: int = 25,
                 commit_timeout_k: int = 5,
                 monitoring_delta: float = None,
                 commit_steps: int = None):
        self.perf = perf
        self.backup = backup
        self.rs = rs
        self.delta_budget_l2 = delta_budget_l2
        self.monitoring_delta = delta_budget_l2 if monitoring_delta is None else monitoring_delta
        self.detection_k = detection_k
        self.recovery_confirm_k = recovery_confirm_k
        self.commit_timeout_k = commit_timeout_k
        # commit_steps: if set, committed PPO lasts this many steps then loops
        # back to Phase 1. If None, commit is permanent (original behaviour).
        self.commit_steps = commit_steps
        self._reset()

    def _reset(self):
        self._phase = self._PPO
        self._consec_unsafe = 0
        self._consec_safe_in_recovery = 0
        self._commit_steps = 0
        self._commit_timer = 0

    def reset_episode(self) -> None:
        self._reset()

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:

        # -- Phase 4: committed PPO (no RS calls) --------------------------------
        if self._phase == self._COMMITTED:
            self._commit_timer += 1
            if self.commit_steps is not None and self._commit_timer >= self.commit_steps:
                # Finite commit: loop back to Phase 1 monitoring
                self._phase = self._PPO
                self._consec_unsafe = 0
                self._commit_timer = 0
            return self.perf.predict(obs_ppo), {
                "allow_perf": 1.0, "p_critical": 0.0,
                "R_rs": float("nan"), "R_exec": float("nan"),
            }

        # -- Phase 1: certified PPO monitoring --------------------------------
        if self._phase == self._PPO:
            pred, p_A_lower, R = self.rs.certify(obs_ppo)
            certified_safe = (pred == 0) and (R >= self.monitoring_delta)

            if not certified_safe:
                self._consec_unsafe += 1
            else:
                self._consec_unsafe = 0

            if self._consec_unsafe >= self.detection_k:
                self._phase = self._ATLA
                self._consec_safe_in_recovery = 0
                self._consec_unsafe = 0
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

        # -- Phase 2: ATLA adaptive recovery ---------------------------------
        if self._phase == self._ATLA:
            pred, p_A_lower, R = self.rs.certify(obs_ppo)
            certified_safe = (pred == 0) and (R >= self.delta_budget_l2)

            if certified_safe:
                self._consec_safe_in_recovery += 1
            else:
                self._consec_safe_in_recovery = 0

            if self._consec_safe_in_recovery >= self.recovery_confirm_k:
                self._phase = self._RS_CHECK
                self._commit_steps = 0

            p_crit = (1.0 - p_A_lower) if pred == 0 else p_A_lower
            return self.backup.predict(obs_atla), {
                "allow_perf": 0.0, "p_critical": p_crit,
                "R_rs": R, "R_exec": float("nan"),
            }

        # -- Phase 3: RS commit check ----------------------------------------
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


class ContinuousSwitcherController:
    """
    Hysteresis-based switcher for arbitrary (multi-burst) attack models.

    No permanent PPO commit.  The controller loops between PPO and ATLA
    indefinitely, using asymmetric entry/exit thresholds to balance false
    alarm rate against detection latency.

    State: PPO (monitoring)
        RS-certify obs each step.
        "not certified safe" = (pred != 0) OR (R < monitoring_delta).
        Maintain alarm counter:
            not safe -> alarm_count += 1
            safe     -> alarm_count = max(0, alarm_count - forgive_decay)
        alarm_count >= K_enter  ->  switch to ATLA, reset alarm_count.

    State: ATLA (recovery)
        RS-certify obs each step.
        "certified safe" = (pred == 0) AND (R >= monitoring_delta).
        Maintain safe counter:
            safe     -> safe_count += 1
            not safe -> safe_count = 0
        safe_count >= K_exit  ->  switch to PPO.

    Hysteresis property: K_enter > K_exit means it's hard to enter ATLA
    (tolerates transient false alarms) but relatively easy to leave once
    attack stops.  Conversely K_enter < K_exit means aggressive detection
    but conservative recovery.  Tune to environment.

    forgive_decay (default 1): how fast alarm_count decays when a safe step
    is seen.  Higher values make detection harder to trigger from scattered
    false alarms; the controller only enters ATLA on genuine bursts of
    consecutive unsafe steps.
    """

    _PPO = "ppo"
    _ATLA = "atla"

    def __init__(self, perf: MuJoCoPerfPolicy, backup: MuJoCoBackupPolicy,
                 rs: VanillaRSSwitcher,
                 delta_budget_l2: float,
                 K_enter: int = 3,
                 K_exit: int = 10,
                 monitoring_delta: float = None,
                 forgive_decay: float = 1.0,
                 exit_window_n: int = None,
                 atla_min_steps: int = 0,
                 ppo_settle_steps: int = 0,
                 transition_blend_k: int = 0):
        self.perf = perf
        self.backup = backup
        self.rs = rs
        self.delta_budget_l2 = delta_budget_l2
        self.monitoring_delta = delta_budget_l2 if monitoring_delta is None else monitoring_delta
        self.K_enter = K_enter
        self.K_exit = K_exit
        self.forgive_decay = forgive_decay
        # exit_window_n: if set, exit ATLA when K_exit out of last exit_window_n
        # steps are certified safe (sliding window). If None, use consecutive count.
        self.exit_window_n = exit_window_n
        # atla_min_steps: minimum steps to stay in ATLA before checking exit condition.
        # Covers the full burst duration before allowing cert-based exit. 0 = no minimum.
        self.atla_min_steps = atla_min_steps
        # ppo_settle_steps: steps after returning from ATLA where alarm counting is
        # suppressed. Prevents immediate re-trigger from post-ATLA transitional
        # states that look adversarial. 0 = no settling period.
        self.ppo_settle_steps = ppo_settle_steps
        # transition_blend_k: after exiting ATLA, linearly blend ATLA→PPO actions
        # over this many steps (alpha=t/K from 0→1). Helps smooth gait mismatch.
        self.transition_blend_k = transition_blend_k
        self._reset()

    def _reset(self):
        self._state = self._PPO
        self._alarm_count = 0.0
        self._safe_count = 0
        self._atla_step_count = 0
        self._ppo_settle_remaining = 0
        self._blend_remaining = 0
        self._exit_window = collections.deque(maxlen=self.exit_window_n) if self.exit_window_n else None

    def reset_episode(self) -> None:
        self._reset()

    def select(self, obs_ppo: np.ndarray, obs_atla: np.ndarray
               ) -> Tuple[np.ndarray, Dict]:

        pred, p_A_lower, R = self.rs.certify(obs_ppo)
        p_crit = (1.0 - p_A_lower) if pred == 0 else p_A_lower

        if self._state == self._PPO:
            certified_safe = (pred == 0) and (R >= self.monitoring_delta)

            if self._ppo_settle_remaining > 0:
                # Settling period after ATLA: suppress alarm counting
                self._ppo_settle_remaining -= 1
            elif not certified_safe:
                self._alarm_count += 1
            else:
                self._alarm_count = max(0.0, self._alarm_count - self.forgive_decay)

            if self._alarm_count >= self.K_enter:
                # Enter ATLA
                self._state = self._ATLA
                self._alarm_count = 0.0
                self._safe_count = 0
                self._atla_step_count = 0
                self._blend_remaining = 0
                return self.backup.predict(obs_atla), {
                    "allow_perf": 0.0, "p_critical": p_crit,
                    "R_rs": R, "R_exec": float("nan"),
                    "phase": "atla_enter",
                }

            # Blend ATLA→PPO actions during transition window
            if self._blend_remaining > 0:
                alpha = float(self.transition_blend_k - self._blend_remaining + 1) / self.transition_blend_k
                ppo_action  = self.perf.predict(obs_ppo)
                atla_action = self.backup.predict(obs_atla)
                action = (1.0 - alpha) * atla_action + alpha * ppo_action
                self._blend_remaining -= 1
                return action, {
                    "allow_perf": alpha, "p_critical": p_crit,
                    "R_rs": R, "R_exec": float("nan"),
                    "phase": "blend",
                }

            return self.perf.predict(obs_ppo), {
                "allow_perf": 1.0, "p_critical": p_crit,
                "R_rs": R, "R_exec": float("nan"),
                "phase": "ppo",
            }

        else:  # self._state == self._ATLA
            certified_safe = (pred == 0) and (R >= self.monitoring_delta)
            self._atla_step_count += 1

            # Only evaluate exit condition after minimum ATLA steps
            if self._atla_step_count < self.atla_min_steps:
                return self.backup.predict(obs_atla), {
                    "allow_perf": 0.0, "p_critical": p_crit,
                    "R_rs": R, "R_exec": float("nan"),
                    "phase": "atla_min",
                }

            if self._exit_window is not None:
                self._exit_window.append(int(certified_safe))
                exit_condition = (len(self._exit_window) == self.exit_window_n
                                  and sum(self._exit_window) >= self.K_exit)
            else:
                if certified_safe:
                    self._safe_count += 1
                else:
                    self._safe_count = 0
                exit_condition = self._safe_count >= self.K_exit

            if exit_condition:
                    # Return to PPO with optional blend and settle period
                    self._state = self._PPO
                    self._alarm_count = 0.0
                    self._safe_count = 0
                    # Settle timer starts AFTER blend completes
                    self._ppo_settle_remaining = self.ppo_settle_steps
                    self._blend_remaining = self.transition_blend_k
                    if self._exit_window is not None:
                        self._exit_window.clear()
                    if self.transition_blend_k > 0:
                        # First blend step: alpha=1/K
                        alpha = 1.0 / self.transition_blend_k
                        ppo_action  = self.perf.predict(obs_ppo)
                        atla_action = self.backup.predict(obs_atla)
                        action = (1.0 - alpha) * atla_action + alpha * ppo_action
                        self._blend_remaining -= 1
                        return action, {
                            "allow_perf": alpha, "p_critical": p_crit,
                            "R_rs": R, "R_exec": R,
                            "phase": "blend",
                        }
                    return self.perf.predict(obs_ppo), {
                        "allow_perf": 1.0, "p_critical": p_crit,
                        "R_rs": R, "R_exec": R,
                        "phase": "ppo_reenter",
                    }

            return self.backup.predict(obs_atla), {
                "allow_perf": 0.0, "p_critical": p_crit,
                "R_rs": R, "R_exec": float("nan"),
                "phase": "atla",
            }


# -- Evaluation loop -----------------------------------------------------------

def _generate_attack_schedule(attack_mode, horizon, burst_k, t_candidate_max,
                              t_candidate_fixed, rng,
                              n_bursts=None, cooldown_k=0):
    """
    Return a boolean array of length `horizon` indicating which steps are
    under attack.

    attack_mode:
      "single"    — one burst per episode (original behavior)
      "multi"     — n_bursts bursts with at least cooldown_k gap between them
      "arbitrary" — each step independently attacked with P(attack) tuned to
                    give ~burst_k total attacked steps spread across episode
    """
    schedule = np.zeros(horizon, dtype=bool)

    if attack_mode == "single":
        T = (t_candidate_fixed if t_candidate_fixed is not None
             else rng.randint(0, t_candidate_max + 1))
        schedule[T:T + burst_k] = True

    elif attack_mode == "multi":
        n = n_bursts if n_bursts is not None else 3
        placed = 0
        t = 0
        while placed < n and t < horizon:
            # Random start within remaining episode
            latest_start = horizon - burst_k
            if t > latest_start:
                break
            T = rng.randint(t, min(t + t_candidate_max, latest_start) + 1)
            schedule[T:T + burst_k] = True
            t = T + burst_k + cooldown_k
            placed += 1

    elif attack_mode == "arbitrary":
        # Bernoulli per step; expected total ≈ burst_k
        p_attack = min(burst_k / max(horizon, 1), 1.0)
        schedule = rng.random(horizon) < p_attack

    return schedule


def evaluate_controller(
    controller,
    perf: MuJoCoPerfPolicy,
    backup: MuJoCoBackupPolicy,
    n_episodes: int = 10,
    seed: int = 0,
    attack: bool = False,
    burst_k: int = 20,
    horizon: int = 1000,
    t_candidate_max: int = 100,
    attack_norm: str = "linf",
    attack_eps: float = None,
    t_candidate_fixed: int = None,
    attack_mode: str = "single",
    n_bursts: int = None,
    cooldown_k: int = 0,
) -> Tuple[List[float], List[Dict]]:
    """
    Run n_episodes with the given controller.

    PPO's custom_env drives the simulation.  At each step, ATLA-normalized obs
    is computed from the raw sim state using ATLA's own read-only ZFilter.

    Attack modes (when attack=True):
      "single"    — one burst per episode (original behavior)
      "multi"     — n_bursts bursts of burst_k steps, cooldown_k gap between
      "arbitrary" — each step independently attacked (Bernoulli, expected total
                    ≈ burst_k steps)
    """
    rng = np.random.RandomState(seed)
    returns: List[float] = []
    logs: List[Dict] = []

    for ep in range(n_episodes):
        obs_ppo = perf.start_episode()
        done = False
        t = 0
        step_logs: List[Dict] = []

        if hasattr(controller, "reset_episode"):
            controller.reset_episode()

        # Pre-generate attack schedule for this episode
        if attack:
            atk_schedule = _generate_attack_schedule(
                attack_mode, horizon, burst_k, t_candidate_max,
                t_candidate_fixed, rng,
                n_bursts=n_bursts, cooldown_k=cooldown_k,
            )
        else:
            atk_schedule = np.zeros(horizon, dtype=bool)

        total_attacked_steps = 0

        while not done and t < horizon:
            # ATLA obs from same raw sim state
            raw = raw_obs_from_sim(perf.custom_env, perf.config)
            obs_atla = backup.normalize(raw)

            # Attack logic
            obs_ctrl = obs_ppo
            if atk_schedule[t] and perf.attack_model is not None:
                eps = attack_eps if attack_eps is not None else perf.eps
                obs_ctrl = opt_attack(perf.attack_model, obs_ppo, eps=eps,
                                      norm=attack_norm)
                total_attacked_steps += 1

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
        R_vals = [l["R_exec"] for l in step_logs if not np.isnan(l["R_exec"])]
        R_mean = float(np.nanmean(R_vals)) if R_vals else float("nan")
        logs.append({"allow_perf": allow_mean, "R_exec": R_mean,
                      "fell": fell, "ep_len": t,
                      "attacked_steps": total_attacked_steps})

    return returns, logs
