"""
Structured Logging - OpenEnv compliant logging format

Format:
[START] task=<task> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
[END] success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import json
from typing import List, Optional
from datetime import datetime


class StructuredLogger:
    """Structured logging for evaluation runs."""
    
    def __init__(self, model_name: str = "unknown", task_name: str = "pcb_triage"):
        self.model_name = model_name
        self.task_name = task_name
        self.steps = []
        self.start_time = None
        self.end_time = None
    
    def start(self, env_name: str = "PCBEnv") -> str:
        """Log start of evaluation."""
        self.start_time = datetime.now().isoformat()
        line = f"[START] task={self.task_name} env={env_name} model={self.model_name} timestamp={self.start_time}"
        print(line)
        return line
    
    def step(
        self,
        step_num: int,
        action: str,
        reward: float,
        done: bool = False,
        error: Optional[str] = None,
    ) -> str:
        """Log a single step."""
        error_str = f'"{error}"' if error else "null"
        line = f"[STEP] step={step_num} action={action} reward={reward:.2f} done={done} error={error_str}"
        print(line)
        self.steps.append({
            "step": step_num,
            "action": action,
            "reward": reward,
            "done": done,
            "error": error,
        })
        return line
    
    def end(self, success: bool, total_steps: int, score: float, rewards: List[float]) -> str:
        """Log end of evaluation."""
        self.end_time = datetime.now().isoformat()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        line = f"[END] success={success} steps={total_steps} score={score:.4f} rewards=[{rewards_str}] timestamp={self.end_time}"
        print(line)
        return line
    
    def to_dict(self) -> dict:
        """Export log as structured dict."""
        return {
            "model": self.model_name,
            "task": self.task_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps": self.steps,
        }
    
    def to_json(self) -> str:
        """Export log as JSON string."""
        return json.dumps(self.to_dict(), indent=2)
