"""
PCB Defect Triage Environment

OpenEnv-compliant environment for PCB defect triage decision-making.
Simulates a factory queue with defective boards requiring routing decisions.
"""

import random
from typing import Tuple, Dict, Any
from .models import PCBObservation, PCBAction, PCBReward


class PCBEnv:
    """
    PCB Defect Triage Environment.
    
    An OpenEnv environment where agents learn to make optimal routing decisions
    for defective PCBs to maximize throughput and cost-effectiveness.
    
    Args:
        batch_size: Number of PCBs to process per episode (default: 50)
        max_slots: Maximum parallel repair slots available (default: 5)
    """

    def __init__(self, batch_size: int = 50, max_slots: int = 5):
        self.batch_size = batch_size
        self.max_slots = max_slots
        self.queue = []
        self.slots = 0
        self.pcbs = []
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0

    def reset(self) -> PCBObservation:
        """
        Reset environment for a new episode.
        
        Returns:
            Initial observation state
        """
        self.pcbs = [self.generate_pcb() for _ in range(self.batch_size)]
        self.queue = []
        self.slots = 0
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        return self.state()

    def generate_pcb(self) -> Dict[str, Any]:
        """
        Generate a random PCB with defect characteristics.
        
        Returns:
            Dictionary with PCB properties
        """
        defect_types = [
            "none",
            "missing_component",
            "solder_bridge",
            "misalignment",
            "short_circuit",
        ]
        return {
            "defect_type": random.choice(defect_types),
            "criticality_score": round(random.random(), 2),
            "component_cost": round(random.uniform(5, 50), 2),
            "inspection_confidence": round(random.random(), 2),
        }

    def step(self, action: PCBAction) -> Tuple[PCBObservation, PCBReward, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: PCBAction specifying the triage decision
            
        Returns:
            Tuple of (observation, reward, done, info_dict)
            
        Raises:
            RuntimeError: If episode is already done
        """
        if self.done:
            raise RuntimeError("Episode already done. Call reset() to start a new episode.")

        pcb = self.pcbs[self.step_count]
        reward = self.compute_reward(pcb, action.action)

        # Update factory slots based on routing decision
        if action.action.startswith("ROUTE") and self.slots < self.max_slots:
            self.slots += 1
            self.queue.append(pcb)
        elif action.action == "WAIT":
            # Penalty for unnecessary waiting
            reward -= 0.1

        # Track episode progress
        self.episode_reward += reward
        self.step_count += 1
        
        if self.step_count >= self.batch_size:
            self.done = True

        next_obs = self.state() if not self.done else None
        return next_obs, PCBReward(reward=reward), self.done, {
            "action": action.action,
            "pcb_defect": pcb["defect_type"],
            "queue_size": len(self.queue),
            "slots_used": self.slots,
        }

    def compute_reward(self, pcb: Dict, action: str) -> float:
        """
        Compute reward based on PCB state and action taken.
        
        Reward structure:
        - +1.0: Correct diagnosis (defect → action match)
        - +0.5: Sent to diagnostics (safe fallback)
        - 0.0: Incorrect action
        - -0.1: Waiting unnecessarily
        
        Args:
            pcb: PCB board state
            action: Action taken
            
        Returns:
            Reward scalar
        """
        # Perfect matches: +1.0
        if pcb["defect_type"] == "missing_component" and action == "SCRAP":
            return 1.0
        elif pcb["defect_type"] == "solder_bridge" and action == "ROUTE_SOLDERING":
            return 1.0
        elif pcb["defect_type"] == "short_circuit" and action == "ROUTE_COMPONENT_REPLACEMENT":
            return 1.0
        elif pcb["defect_type"] == "misalignment" and action == "ROUTE_DIAGNOSTICS":
            return 0.7
        elif pcb["defect_type"] == "none" and action == "PASS":
            return 1.0
        
        # Safe fallback: +0.5
        elif action == "ROUTE_DIAGNOSTICS":
            return 0.5
        
        # Incorrect action: 0.0
        else:
            return 0.0

    def state(self) -> PCBObservation:
        """
        Get current observation state.
        
        Returns:
            PCBObservation for the current PCB in queue
            
        Raises:
            RuntimeError: If querying state after episode is done
        """
        if self.done:
            return None

        pcb = self.pcbs[self.step_count]
        return PCBObservation(
            defect_type=pcb["defect_type"],
            criticality_score=pcb["criticality_score"],
            component_cost=pcb["component_cost"],
            inspection_confidence=pcb["inspection_confidence"],
            queue_length=len(self.queue),
            available_slots=self.max_slots - self.slots,
        )

    def get_episode_score(self) -> float:
        """
        Get average reward for the episode.
        
        Returns:
            Average reward per step
        """
        if self.step_count == 0:
            return 0.0
        return self.episode_reward / self.step_count
