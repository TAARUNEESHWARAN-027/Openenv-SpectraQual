"""
models.py — SpectraQual Typed Pydantic Models
OpenEnv spec requires: typed Observation, Action, Reward models.
"""

from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------
# PCB OBSERVATION
# ---------------------------
class PCBObservation(BaseModel):
    """Observation returned after each reset() or step()."""

    board_id: str = Field(..., description="Unique board identifier, e.g. SQ-4321")
    defect_type: Literal[
        "none", "missing_component", "solder_bridge", "short_circuit"
    ] = Field(..., description="Type of defect detected on the PCB")
    component_cost: float = Field(
        ..., ge=10.0, le=200.0, description="Replacement cost of damaged component in ₹"
    )
    criticality: float = Field(
        ..., ge=0.1, le=1.0, description="Risk score — higher means more critical circuit"
    )
    slots_free: int = Field(
        ..., ge=0, description="Number of soldering slots currently available"
    )
    slots_state: List[int] = Field(
        ..., description="Remaining time units for each soldering slot (0=free)"
    )
    is_anomaly: bool = Field(
        False, description="True if this board exhibits rare/unusual characteristics"
    )
    anomaly_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Anomaly confidence (0=normal, 1=highly anomalous)"
    )
    step: int = Field(..., ge=0, description="Current step number in the episode")
    task_id: str = Field(..., description="ID of the active task")
    valid_actions: List[str] = Field(
        ..., description="List of valid actions for this observation"
    )

    # --- Real-time metrics ---
    rolling_accuracy: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of correct decisions so far"
    )
    throughput: float = Field(
        0.0, ge=0.0, description="Boards processed per step so far"
    )
    cumulative_reward: float = Field(
        0.0, description="Cumulative normalized reward so far in this episode"
    )


# ---------------------------
# PCB ACTION
# ---------------------------
class PCBAction(BaseModel):
    """Action submitted by an agent to the environment."""

    action: Literal[
        "PASS",
        "SCRAP",
        "ROUTE_COMPONENT_REPLACEMENT",
        "ROUTE_SOLDERING",
        "ROUTE_DIAGNOSTICS",
        "WAIT",
    ] = Field(..., description="Decision made for the current PCB")


# ---------------------------
# REWARD COMPONENTS
# ---------------------------
class RewardComponents(BaseModel):
    """Decomposed reward signal for transparency and debugging."""

    defect_reward: float = Field(
        ..., description="Score for handling the defect correctly (0.0–1.0)"
    )
    cost_efficiency: float = Field(
        ..., description="Economic value retained vs. lost (0.0–1.0)"
    )
    queue_penalty: float = Field(
        ..., description="Penalty for creating factory bottlenecks (0.0–1.0, lower is worse)"
    )
    criticality_factor: float = Field(
        ..., description="Risk-adjusted modifier based on criticality (0.0–1.0)"
    )
    anomaly_bonus: float = Field(
        0.0, description="Bonus for correctly flagging/handling anomalous board (0.0–1.0)"
    )
    total_raw: float = Field(
        ..., description="Weighted sum of all components before normalization"
    )
    normalized: float = Field(
        ..., ge=0.0, le=1.0, description="Final normalized reward in [0.0, 1.0]"
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of why this reward was given"
    )


# ---------------------------
# STEP RESULT
# ---------------------------
class StepResult(BaseModel):
    """Full result returned by step() and reset()."""

    observation: PCBObservation
    reward: float = Field(
        0.0, ge=0.0, le=1.0, description="Normalized reward for this step [0.0, 1.0]"
    )
    reward_components: Optional[RewardComponents] = Field(
        None, description="Detailed breakdown of reward components"
    )
    done: bool = Field(..., description="True if the episode has ended")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional diagnostic info"
    )


# ---------------------------
# TASK RESULT (for graders)
# ---------------------------
class TaskResult(BaseModel):
    """Summary of a completed task run, consumed by graders."""

    task_id: str
    total_steps: int
    rewards: List[float]                  # per-step normalized rewards
    correct_decisions: int
    total_decisions: int
    bottleneck_count: int                  # times queue was maxed out
    anomaly_total: int                     # how many anomaly boards appeared
    anomaly_flagged: int                   # how many the agent correctly flagged
    cumulative_raw_reward: float
    max_possible_raw: float
    final_score: float = 0.0              # filled by grader
