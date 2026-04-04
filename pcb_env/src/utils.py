"""
PCB Environment Utilities

Helper functions for state vectorization, action encoding, and data processing.
"""

import numpy as np
from typing import List, Dict, Tuple
from .models import PCBObservation


# Action space definition
ACTION_SPACE = [
    "PASS",
    "SCRAP",
    "ROUTE_COMPONENT_REPLACEMENT",
    "ROUTE_SOLDERING",
    "ROUTE_DIAGNOSTICS",
    "WAIT",
]

# Defect type encoding
DEFECT_ENCODING = {
    "none": 0,
    "missing_component": 1,
    "solder_bridge": 2,
    "misalignment": 3,
    "short_circuit": 4,
}

# Reverse mappings
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTION_SPACE)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}


def observation_to_vector(obs: PCBObservation) -> np.ndarray:
    """
    Convert PCBObservation to numeric vector for RL agents.
    
    Output format:
        [defect_type_encoding, criticality_score, component_cost,
         inspection_confidence, queue_length, available_slots]
    
    Args:
        obs: PCBObservation instance
        
    Returns:
        Numpy array of shape (6,)
    """
    defect_idx = DEFECT_ENCODING.get(obs.defect_type, 0)
    
    state_vector = np.array([
        float(defect_idx),
        obs.criticality_score,
        obs.component_cost / 50.0,  # Normalize to [0, 1]
        obs.inspection_confidence,
        float(obs.queue_length),
        float(obs.available_slots),
    ], dtype=np.float32)
    
    return state_vector


def action_to_index(action: str) -> int:
    """
    Convert action string to index.
    
    Args:
        action: Action string
        
    Returns:
        Action index (0-5)
    """
    return ACTION_TO_IDX.get(action, 5)  # Default to WAIT


def index_to_action(idx: int) -> str:
    """
    Convert action index back to string.
    
    Args:
        idx: Action index
        
    Returns:
        Action string
    """
    return IDX_TO_ACTION.get(idx, "WAIT")


def vectorize_batch(observations: List[PCBObservation]) -> np.ndarray:
    """
    Vectorize a batch of observations.
    
    Args:
        observations: List of PCBObservation instances
        
    Returns:
        Numpy array of shape (batch_size, 6)
    """
    vectors = [observation_to_vector(obs) for obs in observations]
    return np.array(vectors, dtype=np.float32)


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Used for policy gradient methods (PPO, A3C).
    
    Args:
        rewards: List of step rewards
        values: List of value estimates
        gamma: Discount factor
        lambda_: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    returns = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    return advantages, returns


def normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize array to zero mean, unit variance.
    
    Args:
        arr: Numpy array to normalize
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized array
    """
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / (std + eps)


if __name__ == "__main__":
    # Test utils
    from .models import PCBObservation
    
    obs = PCBObservation(
        defect_type="solder_bridge",
        criticality_score=0.85,
        component_cost=25.0,
        inspection_confidence=0.92,
        queue_length=3,
        available_slots=2,
    )
    
    vec = observation_to_vector(obs)
    print("Observation Vector:", vec)
    print("Action to Index (ROUTE_SOLDERING):", action_to_index("ROUTE_SOLDERING"))
    print("Index to Action (3):", index_to_action(3))
