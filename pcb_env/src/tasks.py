"""
PCB Triage Tasks - Easy → Medium → Hard

Pre-defined tasks for testing and grading agent performance at different difficulty levels.
"""

from typing import Dict
from .env import PCBEnv
from .models import PCBAction


def easy_task(env: PCBEnv) -> float:
    """
    Easy Task: Single-step perfect decision for first PCB.
    
    Tests:
        - Binary decision (PASS vs SCRAP)
        - Reward: +1.0 for correct action, 0.0 otherwise
    
    Args:
        env: PCBEnv instance
        
    Returns:
        Reward for the action taken
    """
    obs = env.reset()
    pcb = env.pcbs[0]
    
    # Simple rule: SCRAP missing components, PASS others
    action = "SCRAP" if pcb["defect_type"] == "missing_component" else "PASS"
    
    _, reward, _, _ = env.step(PCBAction(action=action))
    return reward.reward


def medium_task(env: PCBEnv) -> float:
    """
    Medium Task: Routing decision based on defect type.
    
    Tests:
        - Multi-way routing decision (4+ actions)
        - Reward shaping with routing penalties
        - Partial credit for diagnostics
    
    Args:
        env: PCBEnv instance
        
    Returns:
        Average reward for actions taken
    """
    obs = env.reset()
    total_reward = 0.0
    
    for i in range(min(5, env.batch_size)):  # Test on first 5 boards
        pcb = env.pcbs[i]
        
        # Defect-specific routing
        if pcb["defect_type"] == "solder_bridge":
            action = "ROUTE_SOLDERING"
        elif pcb["defect_type"] == "missing_component":
            action = "SCRAP"
        elif pcb["defect_type"] == "none":
            action = "PASS"
        else:
            action = "ROUTE_DIAGNOSTICS"
        
        _, reward, _, _ = env.step(PCBAction(action=action))
        total_reward += reward.reward
    
    return total_reward / 5


def hard_task(env: PCBEnv) -> float:
    """
    Hard Task: Full batch processing with optimal routing.
    
    Tests:
        - Complete episode management (full batch size)
        - Sequential decision quality
        - Resource constraint awareness (slots, queue)
        - Overall policy performance
    
    Args:
        env: PCBEnv instance
        
    Returns:
        Final episode score (average reward)
    """
    obs = env.reset()
    total_reward = 0.0
    step_count = 0
    
    while not env.done:
        pcb = env.pcbs[env.step_count]
        
        # Optimal routing logic
        if pcb["defect_type"] == "missing_component":
            action = "SCRAP"
        elif pcb["defect_type"] == "solder_bridge":
            action = "ROUTE_SOLDERING"
        elif pcb["defect_type"] == "short_circuit":
            action = "ROUTE_COMPONENT_REPLACEMENT"
        elif pcb["defect_type"] == "none":
            action = "PASS"
        else:
            # Fallback for unknown defects
            action = "ROUTE_DIAGNOSTICS"
        
        obs, reward, done, info = env.step(PCBAction(action=action))
        total_reward += reward.reward
        step_count += 1
    
    final_score = total_reward / env.batch_size
    return final_score


def run_all_tasks(env: PCBEnv = None, verbose: bool = True) -> Dict[str, float]:
    """
    Run all tasks and return scores.
    
    Args:
        env: Environment instance (creates new if None)
        verbose: Print results to stdout
        
    Returns:
        Dictionary with task scores
    """
    if env is None:
        env = PCBEnv()
    
    results = {}
    
    # Easy task
    easy_score = easy_task(env)
    results["easy"] = easy_score
    if verbose:
        print(f"[EASY] Score: {easy_score:.4f}")
    
    # Medium task
    medium_score = medium_task(env)
    results["medium"] = medium_score
    if verbose:
        print(f"[MEDIUM] Score: {medium_score:.4f}")
    
    # Hard task
    hard_score = hard_task(env)
    results["hard"] = hard_score
    if verbose:
        print(f"[HARD] Score: {hard_score:.4f}")
    
    if verbose:
        avg_score = (easy_score + medium_score + hard_score) / 3
        print(f"[AVERAGE] Score: {avg_score:.4f}")
    
    return results


if __name__ == "__main__":
    env = PCBEnv(batch_size=50, max_slots=5)
    run_all_tasks(env, verbose=True)
