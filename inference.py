"""
inference.py — SpectraQual OpenEnv Baseline Inference Script

Runs an LLM agent against all 3 SpectraQual tasks and emits structured logs.

Environment variables (set before running):
    API_BASE_URL   The LLM API endpoint  (default: https://openrouter.ai/api/v1)
    MODEL_NAME     Model identifier      (default: meta-llama/llama-3.3-70b-instruct)
    HF_TOKEN       Your Hugging Face / API key (required in production)

Usage:
    export HF_TOKEN="hf_xxx..."
    python inference.py

Output format:
    [START] task=<id> env=SpectraQual model=<model>
    [STEP]  step=<n> action=<A> reward=<r> done=<bool> error=<null|msg>
    [END]   success=<bool> steps=<n> score=<f> rewards=[...]
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import List, Optional

# ── Path setup so we can import from src/ ──────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from openai import OpenAI
from env   import SpectraQualEnv
from models import PCBAction, StepResult
from config import (
    ACTIONS,
    VALID_ACTIONS,
    MAX_STEPS_PER_TASK,
    SUCCESS_SCORE_THRESHOLD,
    TEMPERATURE,
    MAX_TOKENS,
    TASKS,
)
from tasks import TASK_DESCRIPTIONS, run_task, grade

# ── Environment variables ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/llama-3.3-70b-instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = HF_TOKEN or os.getenv("OPENAI_API_KEY", "no-key-set")

# Optional: if you use from_docker_image() style containerized env
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK   = "SpectraQual"
TASK_IDS    = ["task_easy", "task_medium", "task_hard"]

# ── System prompt for the LLM ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a PCB quality-control triage agent.
You will receive information about a printed circuit board (PCB) including its defect type,
component cost, criticality score, and available factory soldering slots.

You must choose exactly ONE action from the allowed list.
Respond with ONLY the action name — no explanation, no extra text, no punctuation.

Action meanings:
- PASS                       → Board has no defect; clear it.
- SCRAP                      → Board is too damaged or high-risk; discard it.
- ROUTE_COMPONENT_REPLACEMENT → Board has a missing component; route to repair.
- ROUTE_SOLDERING             → Board has a solder bridge; send to soldering station.
- ROUTE_DIAGNOSTICS           → Board has an ambiguous fault; send for investigation.
- WAIT                        → No soldering slot available; hold the board.

Rules:
- For defect_type=none, you MUST respond PASS.
- For defect_type=missing_component, choose ROUTE_COMPONENT_REPLACEMENT or SCRAP.
- For defect_type=solder_bridge, choose ROUTE_SOLDERING, WAIT, or SCRAP.
- For defect_type=short_circuit, choose SCRAP or ROUTE_DIAGNOSTICS.
- If slots_free=0 and action=ROUTE_SOLDERING would apply, prefer WAIT instead.

Respond with only one word. Example: ROUTE_SOLDERING"""


# ── Prompt builder ─────────────────────────────────────────────────────────
def build_user_prompt(
    obs,
    step: int,
    last_reward: float,
    history: List[str],
) -> str:
    history_txt = "\n".join(history[-5:]) if history else "None"
    anomaly_txt = f"⚠️ ANOMALY DETECTED (score={obs.anomaly_score:.2f})" if obs.is_anomaly else "Normal"
    return f"""=== PCB TRIAGE — Step {step} ===
Board ID:       {obs.board_id}
Defect Type:    {obs.defect_type}
Component Cost: ₹{obs.component_cost:.2f}
Criticality:    {obs.criticality:.2f}
Slots Free:     {obs.slots_free} / {len(obs.slots_state)}
Slot State:     {obs.slots_state}
Anomaly:        {anomaly_txt}

Valid Actions:  {", ".join(obs.valid_actions)}

Last Reward:    {last_reward:.4f}
Cumulative:     {obs.cumulative_reward:.4f}
Accuracy:       {obs.rolling_accuracy:.2%}

Recent History:
{history_txt}

Choose exactly one action from: {", ".join(obs.valid_actions)}"""


# ── Structured log helpers ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = "null" if error is None else f'"{error}"'
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ──────────────────────────────────────────────────────────────
def get_llm_action(
    client: OpenAI,
    obs,
    step: int,
    last_reward: float,
    history: List[str],
) -> str:
    """Ask the LLM for a triage action. Falls back to SCRAP on any error."""
    prompt = build_user_prompt(obs, step, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip().upper()

        # Validate: pick first word that matches a known action
        for candidate in raw.split():
            candidate = candidate.strip(".,;:!?\"'")
            if candidate in ACTIONS:
                return candidate

        # Fallback: try to find partial match
        for action in ACTIONS:
            if action in raw:
                return action

        print(f"[DEBUG] Unexpected model output: {raw!r}", flush=True)
        return "SCRAP"

    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return "SCRAP"


# ── Single task runner ─────────────────────────────────────────────────────
def run_task_inference(client: OpenAI, task_id: str) -> tuple[bool, int, float, List[float]]:
    """
    Run the LLM agent against one task.
    Returns (success, steps_taken, score, rewards_list).
    """
    cfg         = TASKS[task_id]
    max_steps   = min(cfg["n_boards"] + 5, MAX_STEPS_PER_TASK)
    total_reward_cap = cfg["n_boards"] * 1.0   # max possible (1.0 per step)

    env          = SpectraQualEnv(task_id=task_id)
    history:    List[str]  = []
    rewards:    List[float] = []
    action_log: List[str]  = []
    steps_taken  = 0
    score        = 0.0
    success      = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        obs         = result.observation
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Get action from LLM
            action_str = get_llm_action(client, obs, step, last_reward, history)
            action_log.append(action_str)

            error = None
            try:
                result = env.step(PCBAction(action=action_str))
            except Exception as e:
                error = str(e)
                result = env.step(PCBAction(action="SCRAP"))

            obs         = result.observation
            reward      = result.reward
            done        = result.done
            last_reward = reward

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str!r} → reward={reward:.4f}"
            )

            if done:
                break

        # Score = average normalized reward across all steps
        score = sum(rewards) / max(len(rewards), 1)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task runner error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[DEBUG] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME   = {MODEL_NAME}",   flush=True)
    print(f"[DEBUG] HF_TOKEN     = {'SET' if HF_TOKEN else 'NOT SET (using OPENAI_API_KEY fallback)'}", flush=True)
    print("", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: List[float] = []

    for task_id in TASK_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Starting {task_id} | {TASK_DESCRIPTIONS[task_id][:80]}...", flush=True)
        print(f"{'='*60}\n", flush=True)

        success, steps, score, rewards = run_task_inference(client, task_id)
        all_scores.append(score)

        print(f"\n[DEBUG] {task_id} complete — score={score:.4f} success={success}\n", flush=True)
        time.sleep(1)   # brief pause between tasks

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"[SUMMARY] Overall score={overall:.4f}", flush=True)
    print(f"[SUMMARY] Per-task: { {tid: round(s, 4) for tid, s in zip(TASK_IDS, all_scores)} }", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
