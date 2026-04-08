"""
reward.py — SpectraQual Multi-Component Normalized Reward
Replaces duplicated logic in env.py and old reward.py.

Reward is decomposed into 5 components and normalized to [0.0, 1.0].
This gives the agent a rich, non-sparse signal at every step.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List

from config import (
    REWARD_WEIGHT_DEFECT,
    REWARD_WEIGHT_COST,
    REWARD_WEIGHT_QUEUE,
    REWARD_WEIGHT_CRITICALITY,
    REWARD_WEIGHT_ANOMALY,
    COMPONENT_COST_MIN,
    COMPONENT_COST_MAX,
    ANOMALY_COST_THRESHOLD,
    ANOMALY_CRITICALITY_THRESHOLD,
)
from models import RewardComponents


# ---------------------------
# NORMALIZATION HELPERS
# ---------------------------
def _sigmoid_normalize(x: float, scale: float = 0.025) -> float:
    """Sigmoid-based normalization: output is always in (0, 1)."""
    return 1.0 / (1.0 + math.exp(-scale * x))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _cost_fraction(cost: float) -> float:
    """Normalize cost into [0, 1] range."""
    return (cost - COMPONENT_COST_MIN) / (COMPONENT_COST_MAX - COMPONENT_COST_MIN)


# ---------------------------
# ANOMALY DETECTION
# ---------------------------
def detect_anomaly(pcb: Dict[str, Any]) -> tuple[bool, float]:
    """
    Flag a board as an anomaly if it has extreme cost AND high criticality.
    Returns (is_anomaly, anomaly_score 0.0–1.0).
    """
    cost_flag     = pcb["component_cost"] >= ANOMALY_COST_THRESHOLD
    critical_flag = pcb["criticality"] >= ANOMALY_CRITICALITY_THRESHOLD

    if cost_flag and critical_flag:
        # Combine both signals into a confidence score
        cost_score     = _cost_fraction(pcb["component_cost"])
        critical_score = pcb["criticality"]
        anomaly_score  = _clamp(0.5 * cost_score + 0.5 * critical_score)
        return True, anomaly_score

    # Partial anomaly: one signal strong
    if cost_flag or critical_flag:
        score = _cost_fraction(pcb["component_cost"]) * 0.4 + pcb["criticality"] * 0.3
        return False, _clamp(score)

    return False, 0.0


# ---------------------------
# COMPONENT 1 — DEFECT REWARD
# ---------------------------
def _defect_component(defect: str, action: str) -> tuple[float, str]:
    """
    Score the correctness of the action given the defect type.
    Returns (raw_score 0.0–1.0, explanation_fragment)
    """
    mapping = {
        ("none",               "PASS"):                          (1.00, "Correct PASS on clean board"),
        ("none",               "SCRAP"):                         (0.00, "Wasteful SCRAP on clean board"),
        ("missing_component",  "ROUTE_COMPONENT_REPLACEMENT"):   (1.00, "Optimal route for missing component"),
        ("missing_component",  "SCRAP"):                         (0.30, "Suboptimal SCRAP — value lost"),
        ("solder_bridge",      "ROUTE_SOLDERING"):               (1.00, "Correct soldering route"),
        ("solder_bridge",      "WAIT"):                          (0.40, "WAIT acceptable — preserves board"),
        ("solder_bridge",      "SCRAP"):                         (0.10, "Poor choice — solder bridge is repairable"),
        ("short_circuit",      "SCRAP"):                         (1.00, "Correct SCRAP for high-risk short circuit"),
        ("short_circuit",      "ROUTE_DIAGNOSTICS"):             (0.80, "Diagnostics acceptable for low-risk short"),
        ("short_circuit",      "PASS"):                          (0.00, "Dangerous PASS on short circuit"),
    }
    key = (defect, action)
    if key in mapping:
        score, expl = mapping[key]
        return score, expl
    # Any other invalid combination
    return 0.05, f"Invalid action '{action}' for defect '{defect}'"


# ---------------------------
# COMPONENT 2 — COST EFFICIENCY
# ---------------------------
def _cost_component(defect: str, action: str, cost: float) -> tuple[float, str]:
    """
    Measure economic efficiency of the decision.
    Returns (score 0.0–1.0, explanation_fragment)
    """
    cf = _cost_fraction(cost)

    if defect == "none":
        return (1.0, "No cost involved in PASS") if action == "PASS" else (0.5, "Unnecessary action cost")

    if defect == "missing_component":
        if action == "ROUTE_COMPONENT_REPLACEMENT":
            # High-cost boards benefit more from repair
            return (_clamp(0.5 + 0.5 * cf), f"Repair recovers {cf:.0%} of component value")
        else:  # SCRAP
            # Scrapping expensive boards wastes value
            return (_clamp(1.0 - cf), f"Scrap wastes {cf:.0%} of component value")

    if defect == "solder_bridge":
        if action == "ROUTE_SOLDERING":
            return (_clamp(0.6 + 0.3 * cf), "Soldering route recovers board value")
        elif action == "WAIT":
            return (0.45, "WAIT preserves board but delays throughput")
        else:  # SCRAP
            return (_clamp(0.5 - 0.4 * cf), "Scrapping repairable board is costly")

    if defect == "short_circuit":
        if action == "SCRAP":
            return (0.80, "Scrapping avoids downstream failure cost")
        elif action == "ROUTE_DIAGNOSTICS":
            return (0.70, "Diagnostics adds some cost but recovers revenue")
        else:
            return (0.10, "Wrong action risks high downstream failure penalty")

    return (0.3, "Unknown defect/action combination")


# ---------------------------
# COMPONENT 3 — QUEUE PENALTY
# ---------------------------
def _queue_component(action: str, slots_state: List[int]) -> tuple[float, str]:
    """
    Penalize bottleneck creation. Returns (score 0.0–1.0, explanation_fragment).
    High score = no queue problem. Low score = bad queue usage.
    """
    free_slots = slots_state.count(0)
    total_slots = len(slots_state)

    if action == "ROUTE_SOLDERING":
        if free_slots > 0:
            utilization = 1.0 - (free_slots - 1) / total_slots
            return (_clamp(0.6 + 0.4 * utilization),
                    f"Soldering assigned to free slot ({free_slots - 1} remaining)")
        else:
            # All slots full → bottleneck
            return (0.0, "BOTTLENECK: all soldering slots occupied")

    if action == "WAIT":
        if free_slots == 0:
            return (0.55, "WAIT appropriate — no slot available")
        else:
            return (0.35, "Unnecessary WAIT — slots were available")

    # Non-soldering actions don't stress the queue
    occupancy_ratio = sum(1 for s in slots_state if s > 0) / total_slots
    return (_clamp(1.0 - 0.2 * occupancy_ratio), "No queue impact from this action")


# ---------------------------
# COMPONENT 4 — CRITICALITY
# ---------------------------
def _criticality_component(defect: str, action: str, criticality: float) -> tuple[float, str]:
    """
    Risk-adjust the decision based on board criticality.
    High-criticality wrong decisions are severely penalized.
    """
    # Optimal action scores well regardless of criticality
    optimal = {
        "none":              "PASS",
        "missing_component": "ROUTE_COMPONENT_REPLACEMENT",
        "solder_bridge":     "ROUTE_SOLDERING",
        "short_circuit":     "SCRAP",
    }
    is_optimal = (optimal.get(defect) == action)

    if is_optimal:
        # Reward scales slightly with criticality — making the right call on risky boards is harder
        return (_clamp(0.7 + 0.3 * criticality), f"Correct action on criticality={criticality:.2f} board")

    if defect == "short_circuit" and action not in ("SCRAP", "ROUTE_DIAGNOSTICS"):
        # Dangerous wrong action on high-criticality board
        penalty = criticality
        return (_clamp(1.0 - penalty), f"Risky action on critical short_circuit board (criticality={criticality:.2f})")

    # Sub-optimal but not dangerous
    return (_clamp(0.5 - 0.2 * criticality), f"Sub-optimal action with criticality={criticality:.2f}")


# ---------------------------
# COMPONENT 5 — ANOMALY BONUS
# ---------------------------
def _anomaly_component(is_anomaly: bool, action: str, defect: str) -> tuple[float, str]:
    """
    Bonus for handling anomalous boards correctly.
    For inference.py the LLM can't explicitly 'flag' anomalies, so we reward
    it for choosing the safest action on anomaly boards.
    """
    if not is_anomaly:
        return (0.5, "Normal board — no anomaly bonus/penalty")

    # Best safe action on anomaly board
    safe_actions = {
        "none":              "PASS",
        "missing_component": "ROUTE_COMPONENT_REPLACEMENT",
        "solder_bridge":     "ROUTE_SOLDERING",
        "short_circuit":     "SCRAP",
    }
    if action == safe_actions.get(defect):
        return (1.0, "Correct safe action on anomaly board — BONUS")
    elif action == "SCRAP":
        return (0.6, "Conservative SCRAP on anomaly board")
    else:
        return (0.1, "Risky action on anomaly board — PENALTY")


# ---------------------------
# MASTER REWARD CALCULATOR
# ---------------------------
def calculate_reward(
    pcb: Dict[str, Any],
    action: str,
    slots_state: List[int],
    is_anomaly: bool = False,
) -> RewardComponents:
    """
    Compute multi-component normalized reward for a (pcb, action) pair.

    Args:
        pcb:         dict with defect_type, component_cost, criticality
        action:      one of the 6 valid action strings
        slots_state: list of slot remaining times, e.g. [0, 2, 0]
        is_anomaly:  whether this board was flagged as anomalous

    Returns:
        RewardComponents with individual scores and final normalized reward.
    """
    defect      = pcb["defect_type"]
    cost        = pcb["component_cost"]
    criticality = pcb["criticality"]

    # Compute each component
    d_score, d_expl = _defect_component(defect, action)
    c_score, c_expl = _cost_component(defect, action, cost)
    q_score, q_expl = _queue_component(action, slots_state)
    r_score, r_expl = _criticality_component(defect, action, criticality)
    a_score, a_expl = _anomaly_component(is_anomaly, action, defect)

    # Weighted sum
    raw = (
        REWARD_WEIGHT_DEFECT      * d_score +
        REWARD_WEIGHT_COST        * c_score +
        REWARD_WEIGHT_QUEUE       * q_score +
        REWARD_WEIGHT_CRITICALITY * r_score +
        REWARD_WEIGHT_ANOMALY     * a_score
    )
    normalized = _clamp(raw)

    # Build explanation
    parts = [
        f"[Defect {d_score:.2f}] {d_expl}",
        f"[Cost {c_score:.2f}] {c_expl}",
        f"[Queue {q_score:.2f}] {q_expl}",
        f"[Risk {r_score:.2f}] {r_expl}",
    ]
    if is_anomaly:
        parts.append(f"[Anomaly {a_score:.2f}] {a_expl}")
    explanation = " | ".join(parts)

    return RewardComponents(
        defect_reward=round(d_score, 4),
        cost_efficiency=round(c_score, 4),
        queue_penalty=round(q_score, 4),
        criticality_factor=round(r_score, 4),
        anomaly_bonus=round(a_score, 4),
        total_raw=round(raw, 4),
        normalized=round(normalized, 4),
        explanation=explanation,
    )