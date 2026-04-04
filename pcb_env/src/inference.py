"""
PCB Triage Inference Script - Baseline Agent

Demonstrates environment usage with structured logging for OpenEnv contests.
Uses rule-based heuristics for PCB routing decisions.
"""

import asyncio
import json
from datetime import datetime
from .env import PCBEnv
from .models import PCBAction


class BaselineAgent:
    """Simple rule-based baseline agent for PCB triage."""
    
    def __init__(self):
        self.name = "BaselineAgent_v1"
        self.actions_taken = []
    
    def select_action(self, obs) -> str:
        """
        Select action using rule-based heuristics.
        
        Rules (in priority order):
        1. High criticality + high confidence → route to appropriate specialist
        2. Unknown defects → diagnostics
        3. Perfect boards → pass
        
        Args:
            obs: PCBObservation
            
        Returns:
            Action string
        """
        # If inspection is uncertain, go to diagnostics
        if obs.inspection_confidence < 0.5:
            return "ROUTE_DIAGNOSTICS"
        
        # Defect-specific routing
        if obs.defect_type == "none":
            return "PASS"
        elif obs.defect_type == "missing_component":
            return "SCRAP"
        elif obs.defect_type == "solder_bridge":
            return "ROUTE_SOLDERING"
        elif obs.defect_type == "short_circuit":
            return "ROUTE_COMPONENT_REPLACEMENT"
        elif obs.defect_type == "misalignment":
            return "ROUTE_DIAGNOSTICS"
        else:
            # Unknown defect type
            return "ROUTE_DIAGNOSTICS"


async def main():
    """
    Run baseline inference with structured logging.
    
    Output format:
        [START] Task metadata
        [STEP] Step-by-step execution with actions and rewards
        [END] Final score
    """
    print("=" * 80)
    print("PCB DEFECT TRIAGE - BASELINE INFERENCE")
    print("=" * 80)
    
    # Initialize
    env = PCBEnv(batch_size=50, max_slots=5)
    agent = BaselineAgent()
    
    obs = env.reset()
    
    # Metadata
    timestamp = datetime.now().isoformat()
    print(f"\n[START] Task: PCB Triage Baseline")
    print(f"  Timestamp: {timestamp}")
    print(f"  Agent: {agent.name}")
    print(f"  Batch Size: {env.batch_size}")
    print(f"  Max Slots: {env.max_slots}")
    print(f"  Initial Observation: {obs}\n")
    
    # Execution loop
    step = 0
    total_reward = 0.0
    action_histogram = {}
    
    print("EXECUTION LOG")
    print("-" * 80)
    
    while not env.done:
        step += 1
        
        # Select action
        action_str = agent.select_action(obs)
        action_histogram[action_str] = action_histogram.get(action_str, 0) + 1
        
        # Execute
        obs, reward, done, info = env.step(PCBAction(action=action_str))
        total_reward += reward.reward
        
        # Log step
        print(
            f"[STEP {step:03d}] "
            f"Action: {action_str:30s} | "
            f"Reward: {reward.reward:+.2f} | "
            f"Queue: {info['queue_size']:2d} | "
            f"Slots: {info['slots_used']:2d}"
        )
        
        agent.actions_taken.append({
            "step": step,
            "action": action_str,
            "reward": reward.reward,
            "pcb_defect": info["pcb_defect"],
        })
    
    # Final score
    score = env.get_episode_score()
    print("-" * 80)
    print(f"\n[END] RESULTS")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Average Score: {score:.4f}")
    print(f"  Queue Final Size: {len(env.queue)}")
    print(f"  Slots Used: {env.slots}/{env.max_slots}")
    
    print(f"\nACTION DISTRIBUTION")
    for action, count in sorted(action_histogram.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / step
        print(f"  {action:35s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"BASELINE SCORE: {score:.4f}")
    print("=" * 80)
    
    return score


if __name__ == "__main__":
    final_score = asyncio.run(main())
