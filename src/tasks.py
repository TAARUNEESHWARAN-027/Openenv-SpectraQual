"""
tasks.py — SpectraQual Task Definitions and Programmatic Graders
Each task runs the environment with a fixed seed and scores the agent 0.0–1.0.
Graders are deterministic and reproducible.
"""

from __future__ import annotations
import sys
import os
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TASKS,
    MEDIUM_ECONOMIC_TARGET,
    HARD_ANOMALY_RATE_TARGET,
    SUCCESS_SCORE_THRESHOLD,
)
from models import TaskResult
from env import SpectraQualEnv
from models import PCBAction


# ---------------------------
# TASK RUNNER
# ---------------------------
def run_task(task_id: str, actions: List[str]) -> TaskResult:
    """
    Run a task with a pre-determined list of actions.
    Used by graders to replay an agent's trajectory deterministically.

    Args:
        task_id: one of "task_easy", "task_medium", "task_hard"
        actions:  list of action strings, one per step

    Returns:
        TaskResult with all episode metrics filled in.
    """
    cfg = TASKS[task_id]
    env = SpectraQualEnv(task_id=task_id)
    env.reset()

    rewards:    List[float] = []
    correct     = 0
    total       = 0
    bottlenecks = 0
    anomaly_total   = 0
    anomaly_flagged = 0
    cum_raw     = 0.0

    for i, action_str in enumerate(actions):
        if env._done:
            break

        # Default to SCRAP if action is out of valid range
        valid = env._current_pcb and env._current_pcb.get("defect_type")
        try:
            result = env.step(PCBAction(action=action_str))
        except Exception:
            result = env.step(PCBAction(action="SCRAP"))

        rewards.append(result.reward)
        total += 1
        if result.info.get("is_anomaly"):
            anomaly_total += 1
        if result.reward_components:
            cum_raw += result.reward_components.total_raw
            if result.info.get("is_anomaly") and result.reward_components.anomaly_bonus >= 0.8:
                anomaly_flagged += 1

        if env._is_correct(result.info.get("defect", ""), action_str):
            correct += 1

        bottlenecks = env._bottleneck_cnt

    max_possible_raw = cfg["n_boards"] * 1.0  # max normalized = 1.0 per step

    return TaskResult(
        task_id=task_id,
        total_steps=total,
        rewards=rewards,
        correct_decisions=correct,
        total_decisions=total,
        bottleneck_count=bottlenecks,
        anomaly_total=anomaly_total,
        anomaly_flagged=anomaly_flagged,
        cumulative_raw_reward=cum_raw,
        max_possible_raw=max_possible_raw,
    )


# ---------------------------
# GRADER: TASK EASY
# ---------------------------
def grade_easy(result: TaskResult) -> float:
    """
    Task Easy Grader.
    Objective: Correctly classify all defect types. No slot pressure.
    Scoring: correct_decisions / total_decisions → 0.0–1.0

    Also gives partial credit for near-correct results:
    - 100% correct = 1.0
    - 80% correct  = 0.8
    - 0% correct   = 0.0
    """
    if result.total_decisions == 0:
        return 0.0

    accuracy = result.correct_decisions / result.total_decisions

    # Blend accuracy with average reward for robustness
    avg_reward = sum(result.rewards) / len(result.rewards) if result.rewards else 0.0

    # Weight: 70% accuracy, 30% reward quality
    score = 0.70 * accuracy + 0.30 * avg_reward
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------
# GRADER: TASK MEDIUM
# ---------------------------
def grade_medium(result: TaskResult) -> float:
    """
    Task Medium Grader.
    Objective: Triage 15 boards with 1 slot (queue pressure).
    Scoring: 0.6 * economic_efficiency + 0.4 * bottleneck_avoidance

    - economic_efficiency: avg normalized reward vs target
    - bottleneck_avoidance: 1.0 if no bottlenecks, scales down to 0
    """
    if not result.rewards:
        return 0.0

    avg_reward = sum(result.rewards) / len(result.rewards)

    # Economic efficiency: how close to target (MEDIUM_ECONOMIC_TARGET = 0.50)
    economic_score = min(avg_reward / MEDIUM_ECONOMIC_TARGET, 1.0)

    # Bottleneck avoidance: 0 bottleneck = 1.0, ≥5 = 0.0
    max_tolerable_bottlenecks = 5
    bottleneck_score = max(0.0, 1.0 - result.bottleneck_count / max_tolerable_bottlenecks)

    score = 0.60 * economic_score + 0.40 * bottleneck_score
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------
# GRADER: TASK HARD
# ---------------------------
def grade_hard(result: TaskResult) -> float:
    """
    Task Hard Grader.
    Objective: 20 boards, mixed anomalies, tight slots.
    Scoring: 0.5 * anomaly_score + 0.3 * economic_score + 0.2 * throughput_score

    - anomaly_score:    anomaly_flagged / max(anomaly_total, 1), target ≥ 0.5
    - economic_score:   avg normalized reward
    - throughput_score: boards_processed / total (penalizes WAIT spam)
    """
    if not result.rewards:
        return 0.0

    cfg = TASKS["task_hard"]
    avg_reward = sum(result.rewards) / len(result.rewards)

    # Anomaly score: did the agent handle anomalous boards correctly?
    if result.anomaly_total > 0:
        raw_anomaly = result.anomaly_flagged / result.anomaly_total
    else:
        raw_anomaly = 1.0  # no anomalies → not penalized

    # Scale anomaly score: meeting HARD_ANOMALY_RATE_TARGET = 1.0
    anomaly_score = min(raw_anomaly / HARD_ANOMALY_RATE_TARGET, 1.0)

    # Economic score
    economic_score = avg_reward

    # Throughput: penalize excessive WAIT actions
    throughput_score = min(result.total_decisions / cfg["n_boards"], 1.0)

    score = (
        0.50 * anomaly_score +
        0.30 * economic_score +
        0.20 * throughput_score
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------
# GRADER DISPATCH
# ---------------------------
GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}


def grade(task_id: str, result: TaskResult) -> float:
    """Dispatch to the correct grader for the given task_id."""
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task_id='{task_id}'")
    return GRADERS[task_id](result)


# ---------------------------
# TASK DESCRIPTIONS (for README / inference prompt)
# ---------------------------
TASK_DESCRIPTIONS = {
    "task_easy": (
        "Triage 10 PCBs with no factory slot pressure. "
        "Focus: identify the correct action for each defect type. "
        "Grader: accuracy-weighted reward (70% accuracy + 30% reward quality). "
        "Expected frontier model score: ≥0.85."
    ),
    "task_medium": (
        "Triage 15 PCBs with only 1 active soldering slot. "
        "Focus: manage queue pressure while maintaining economic performance. "
        "Grader: 60% economic efficiency + 40% bottleneck avoidance. "
        "Expected frontier model score: ≥0.65."
    ),
    "task_hard": (
        "Triage 20 PCBs with 25% anomaly rate and tight slot constraints. "
        "Focus: handle extreme-cost/criticality boards safely AND maintain throughput. "
        "Grader: 50% anomaly handling + 30% economic score + 20% throughput. "
        "Expected frontier model score: ≥0.50."
    ),
}


# ---------------------------
# CLI TEST UTILITY
# ---------------------------
if __name__ == "__main__":
    """Quick sanity check: run all 3 tasks with a rule-based agent."""
    from env import SpectraQualEnv, decide_action
    from models import PCBAction

    print("\n=== SpectraQual Task Grader Sanity Check ===\n")

    for tid in ["task_easy", "task_medium", "task_hard"]:
        env = SpectraQualEnv(task_id=tid)
        result_obj = env.reset()
        actions = []

        while not result_obj.done:
            obs = result_obj.observation
            pcb = {
                "defect_type":    obs.defect_type,
                "component_cost": obs.component_cost,
                "criticality":    obs.criticality,
            }
            action_str  = decide_action(pcb)
            actions.append(action_str)
            result_obj  = env.step(PCBAction(action=action_str))

        task_result = run_task(tid, actions)
        score       = grade(tid, task_result)
        print(f"[{tid}] Score: {score:.4f} | Correct: {task_result.correct_decisions}/{task_result.total_decisions} | Bottlenecks: {task_result.bottleneck_count}")

    print("\n=== Done ===")
