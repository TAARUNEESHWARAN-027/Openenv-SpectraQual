"""
Q-Learning Agent - Lightweight RL model for local training

Simple tabular Q-learning agent that can be trained in-browser.
"""

import numpy as np
from typing import Tuple, Optional
from .models import PCBObservation
from .utils import observation_to_vector, index_to_action, action_to_index


class QLearningAgent:
    """Tabular Q-Learning agent."""
    
    def __init__(
        self,
        state_size: int = 6,
        action_size: int = 6,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of actions
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: discretized state → action → Q-value
        # For continuous states, we'll use state hashing or neural network approximation
        self.q_table = {}
        self.episode_rewards = []
    
    def _discretize_state(self, state_vec: np.ndarray) -> tuple:
        """Convert continuous state vector to discrete key for Q-table."""
        # Quantize continuous values to bins
        discrete_state = tuple(np.round(state_vec * 10).astype(int))
        return discrete_state
    
    def select_action(self, obs: PCBObservation, training: bool = True) -> str:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: PCBObservation
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Action string
        """
        state_vec = observation_to_vector(obs)
        state_key = self._discretize_state(state_vec)
        
        # Initialize Q-values for state if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Epsilon-greedy policy
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(self.action_size)
        else:
            # Exploit: best action
            action_idx = np.argmax(self.q_table[state_key])
        
        return index_to_action(action_idx)
    
    def update(
        self,
        obs: PCBObservation,
        action: str,
        reward: float,
        next_obs: Optional[PCBObservation],
        done: bool,
    ):
        """
        Update Q-values based on experience.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation (None if done)
            done: Whether episode is terminated
        """
        state_vec = observation_to_vector(obs)
        state_key = self._discretize_state(state_vec)
        action_idx = action_to_index(action)
        
        # Initialize state if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Get max Q-value for next state
        if done or next_obs is None:
            max_next_q = 0
        else:
            next_state_vec = observation_to_vector(next_obs)
            next_state_key = self._discretize_state(next_state_vec)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update
        current_q = self.q_table[state_key][action_idx]
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q
    
    def end_episode(self, episode_reward: float):
        """Log episode completion and decay epsilon."""
        self.episode_rewards.append(episode_reward)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        if not self.episode_rewards:
            return {"episodes": 0, "avg_reward": 0, "max_reward": 0}
        
        return {
            "episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "epsilon": self.epsilon,
        }
