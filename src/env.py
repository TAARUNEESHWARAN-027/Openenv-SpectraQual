"""
env.py — SpectraQual OpenEnv-Compliant Environment
Implements the full OpenEnv interface: reset() / step() / state()
with seeding, anomaly detection, episode management, and rolling metrics.
"""

from __future__ import annotations
import random
import sys
import os
from typing import Dict, Any, Optional, List

# Allow running from src/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEFECT_TYPES,
    VALID_ACTIONS,
    N_SOLDERING_SLOTS,
    SOLDERING_JOB_DURATION,
    COMPONENT_COST_MIN,
    COMPONENT_COST_MAX,
    CRITICALITY_MIN,
    CRITICALITY_MAX,
    TASKS,
)
from models import PCBObservation, PCBAction, StepResult, RewardComponents
from reward import calculate_reward, detect_anomaly


# ---------------------------
# SPECTRAQUAL ENVIRONMENT
# ---------------------------
class SpectraQualEnv:
    """
    PCB Smart Quality-Control Triage Environment.

    An AI agent processes a stream of printed circuit boards, each with a
    randomly (but reproducibly seeded) assigned defect. The agent must choose
    the optimal triage action given economic constraints and factory slot availability.

    Implements the OpenEnv interface:
        reset()  → StepResult (initial observation)
        step()   → StepResult
        state()  → dict (full internal state)
    """

    def __init__(self, task_id: str = "task_easy", seed: Optional[int] = None):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")

        self.task_cfg   = TASKS[task_id]
        self.task_id    = task_id
        self.seed       = seed if seed is not None else self.task_cfg["seed"]
        self._rng       = random.Random(self.seed)

        # Runtime state (initialized on reset)
        self._slots:          List[int]          = []
        self._step_num:       int                = 0
        self._done:           bool               = True
        self._current_pcb:    Optional[Dict]     = None
        self._correct_count:  int                = 0
        self._total_count:    int                = 0
        self._bottleneck_cnt: int                = 0
        self._anomaly_total:  int                = 0
        self._anomaly_flagged:int                = 0
        self._cumulative_reward: float           = 0.0
        self._reward_history: List[float]        = []
        self._all_rewards:    List[float]        = []

    # ------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------
    def _reset_slots(self) -> None:
        n = self.task_cfg["n_slots"]
        # Fill remaining slots with 0 (free) up to N_SOLDERING_SLOTS
        self._slots = [0] * N_SOLDERING_SLOTS
        # Mark slots beyond the task limit as permanently busy (simulates fewer slots)
        for i in range(n, N_SOLDERING_SLOTS):
            self._slots[i] = 9999  # permanently locked

    def _get_slot_view(self) -> List[int]:
        """Public view: replace 9999 sentinel with -1 for clarity."""
        return [s if s != 9999 else -1 for s in self._slots]

    def _count_free_slots(self) -> int:
        return sum(1 for s in self._slots if s == 0)

    def _tick_slots(self) -> None:
        """Advance factory time: reduce non-locked slot timers by 1."""
        for i in range(len(self._slots)):
            if 0 < self._slots[i] < 9999:
                self._slots[i] -= 1

    def _assign_slot(self) -> bool:
        """Try to assign a soldering job. Returns True if successful."""
        for i in range(len(self._slots)):
            if self._slots[i] == 0:
                self._slots[i] = SOLDERING_JOB_DURATION
                return True
        return False

    def _generate_pcb(self) -> Dict[str, Any]:
        """Generate a random PCB using internal seeded RNG."""
        # Inject anomaly based on task config
        anomaly_roll = self._rng.random()
        anomaly_rate = self.task_cfg.get("anomaly_rate", 0.0)

        if anomaly_rate > 0 and anomaly_roll < anomaly_rate:
            # Force extreme values
            cost        = round(self._rng.uniform(185.0, 200.0), 2)
            criticality = round(self._rng.uniform(0.93, 1.0), 2)
            defect      = self._rng.choice(["missing_component", "short_circuit"])
        else:
            defect      = self._rng.choice(DEFECT_TYPES)
            cost        = round(self._rng.uniform(COMPONENT_COST_MIN, COMPONENT_COST_MAX), 2)
            criticality = round(self._rng.uniform(CRITICALITY_MIN, CRITICALITY_MAX), 2)

        board_id = f"SQ-{self._rng.randint(1000, 9999)}"

        return {
            "board_id":       board_id,
            "defect_type":    defect,
            "component_cost": cost,
            "criticality":    criticality,
        }

    def _is_correct(self, defect: str, action: str) -> bool:
        """Check if action is the single best action for this defect."""
        best = {
            "none":              "PASS",
            "missing_component": "ROUTE_COMPONENT_REPLACEMENT",
            "solder_bridge":     "ROUTE_SOLDERING",
            "short_circuit":     "SCRAP",
        }
        return best.get(defect) == action

    def _build_observation(self, is_anomaly: bool, anomaly_score: float) -> PCBObservation:
        pcb         = self._current_pcb
        defect      = pcb["defect_type"]
        free_slots  = self._count_free_slots()
        slot_view   = self._get_slot_view()
        total       = self._total_count or 1

        return PCBObservation(
            board_id=pcb["board_id"],
            defect_type=defect,
            component_cost=pcb["component_cost"],
            criticality=pcb["criticality"],
            slots_free=free_slots,
            slots_state=slot_view,
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            step=self._step_num,
            task_id=self.task_id,
            valid_actions=VALID_ACTIONS.get(defect, ["SCRAP"]),
            rolling_accuracy=round(self._correct_count / total, 4),
            throughput=round(self._total_count / max(self._step_num, 1), 4),
            cumulative_reward=round(self._cumulative_reward, 4),
        )

    # ------------------------------------------------
    # PUBLIC OPENENV INTERFACE
    # ------------------------------------------------
    def reset(self) -> StepResult:
        """
        Reset the environment to a clean initial state.
        Returns the first observation without a reward.
        """
        self._rng             = random.Random(self.seed)
        self._step_num        = 0
        self._done            = False
        self._correct_count   = 0
        self._total_count     = 0
        self._bottleneck_cnt  = 0
        self._anomaly_total   = 0
        self._anomaly_flagged = 0
        self._cumulative_reward = 0.0
        self._reward_history  = []
        self._all_rewards     = []

        self._reset_slots()
        self._current_pcb = self._generate_pcb()

        is_anomaly, anomaly_score = detect_anomaly(self._current_pcb)
        if is_anomaly:
            self._anomaly_total += 1

        obs = self._build_observation(is_anomaly, anomaly_score)

        return StepResult(
            observation=obs,
            reward=0.0,
            reward_components=None,
            done=False,
            info={"message": "Environment reset. Episode started.", "seed": self.seed},
        )

    def step(self, action: PCBAction) -> StepResult:
        """
        Apply an action to the current board.
        Advances factory state, computes reward, generates next PCB.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        self._step_num  += 1
        self._total_count += 1
        action_str = action.action
        pcb        = self._current_pcb
        defect     = pcb["defect_type"]

        # Check if action is valid (penalize but don't crash)
        valid = VALID_ACTIONS.get(defect, ["SCRAP"])
        if action_str not in valid:
            # Remap invalid action to SCRAP (safe fallback)
            action_str = "SCRAP"

        # Factory tick
        self._tick_slots()

        # Handle soldering slot assignment
        if action_str == "ROUTE_SOLDERING":
            assigned = self._assign_slot()
            if not assigned:
                self._bottleneck_cnt += 1

        # Anomaly detection
        is_anomaly, anomaly_score = detect_anomaly(pcb)
        if is_anomaly:
            self._anomaly_total += 1
            # Track if agent "handled" anomaly correctly (chose optimal action)
            if self._is_correct(defect, action_str):
                self._anomaly_flagged += 1

        # Reward
        rc = calculate_reward(
            pcb=pcb,
            action=action_str,
            slots_state=self._slots,
            is_anomaly=is_anomaly,
        )
        reward = rc.normalized
        self._cumulative_reward += reward
        self._all_rewards.append(reward)
        self._reward_history.append(reward)

        # Accuracy tracking
        if self._is_correct(defect, action_str):
            self._correct_count += 1

        # Episode done?
        max_boards = self.task_cfg["n_boards"]
        done = (self._total_count >= max_boards)
        self._done = done

        # Prepare next PCB (for observation even if done)
        if not done:
            self._current_pcb = self._generate_pcb()
            next_is_anomaly, next_anomaly_score = detect_anomaly(self._current_pcb)
        else:
            # Episode over — reuse last PCB for observation
            next_is_anomaly, next_anomaly_score = is_anomaly, anomaly_score

        obs = self._build_observation(next_is_anomaly, next_anomaly_score)

        return StepResult(
            observation=obs,
            reward=reward,
            reward_components=rc,
            done=done,
            info={
                "action_taken":     action_str,
                "defect":           defect,
                "board_id":         pcb["board_id"],
                "is_anomaly":       is_anomaly,
                "anomaly_score":    round(anomaly_score, 4),
                "bottleneck_count": self._bottleneck_cnt,
                "step":             self._step_num,
                "correct_count":    self._correct_count,
                "total_count":      self._total_count,
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return the full internal environment state as a dict."""
        return {
            "task_id":           self.task_id,
            "seed":              self.seed,
            "step":              self._step_num,
            "done":              self._done,
            "slots":             self._get_slot_view(),
            "free_slots":        self._count_free_slots(),
            "current_pcb":       self._current_pcb,
            "correct_count":     self._correct_count,
            "total_count":       self._total_count,
            "bottleneck_count":  self._bottleneck_cnt,
            "anomaly_total":     self._anomaly_total,
            "anomaly_flagged":   self._anomaly_flagged,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "reward_history":    self._all_rewards,
            "rolling_accuracy":  round(self._correct_count / max(self._total_count, 1), 4),
            "throughput":        round(self._total_count / max(self._step_num, 1), 4),
        }


# ---------------------------
# LEGACY COMPAT (for main.py / train.py / app.py)
# ---------------------------
# The old code imported module-level factory dict + generate_pcb / decide_action etc.
# We keep those here as thin wrappers so existing imports don't break.

_default_env = SpectraQualEnv("task_easy")

factory = {"soldering_slots": _default_env._slots}


def generate_pcb():
    return _default_env._generate_pcb()


def update_factory():
    _default_env._tick_slots()
    factory["soldering_slots"] = _default_env._get_slot_view()


def assign_soldering_job():
    return _default_env._assign_slot()


def decide_action(pcb):
    """Legacy rule-based decision (used by main.py)."""
    from config import VALID_ACTIONS
    defect = pcb["defect_type"]
    cost   = pcb["component_cost"]
    critical = pcb["criticality"]

    if defect == "none":
        return "PASS"
    if defect == "missing_component":
        return "ROUTE_COMPONENT_REPLACEMENT" if cost > 50 else "SCRAP"
    if defect == "solder_bridge":
        return "ROUTE_SOLDERING" if _default_env._count_free_slots() > 0 else "WAIT"
    if defect == "short_circuit":
        return "SCRAP" if critical > 0.7 else "ROUTE_DIAGNOSTICS"
    return "SCRAP"


def calculate_reward_legacy(pcb, decision):
    """Legacy single-float reward (used by train.py)."""
    rc = calculate_reward(
        pcb=pcb,
        action=decision,
        slots_state=_default_env._slots,
        is_anomaly=False,
    )
    # Scale normalized [0,1] back to a range train.py expects
    return (rc.normalized - 0.5) * 200