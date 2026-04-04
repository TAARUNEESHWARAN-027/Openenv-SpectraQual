"""
Training Loop - Train RL agents on the environment
"""

from typing import Callable, Optional, List, Tuple
from .env import PCBEnv
from .models import PCBAction
from .logging_utils import StructuredLogger


def train_agent(
    agent: Callable,  # Agent with select_action() and update() methods
    env: PCBEnv,
    episodes: int = 50,
    model_name: str = "QLearningAgent",
    verbose: bool = True,
) -> Tuple[float, float, List[float]]:
    """
    Train an RL agent on the environment.
    
    Args:
        agent: Agent instance with select_action(obs) and update(obs, action, reward, next_obs, done)
        env: PCBEnv instance
        episodes: Number of episodes to train
        model_name: Name of model for logging
        verbose: Print progress
        
    Returns:
        Tuple of (final_avg_score, max_score, episode_rewards)
    """
    logger = StructuredLogger(model_name=model_name, task_name="training")
    episode_rewards = []
    
    for episode in range(episodes):
        logger.start()
        
        obs = env.reset()
        ep_reward = 0.0
        step = 0
        
        while not env.done:
            step += 1
            
            # Agent selects action
            action_str = agent.select_action(obs, training=True)
            action_obj = PCBAction(action=action_str)
            
            # Execute step
            next_obs, reward, done, info = env.step(action_obj)
            
            # Agent updates
            agent.update(obs, action_str, reward.reward, next_obs, done)
            
            logger.step(step, action_str, reward.reward, done)
            ep_reward += reward.reward
            obs = next_obs
        
        # Log episode end
        score = env.get_episode_score()
        logger.end(success=True, total_steps=step, score=score, rewards=[score] * step)
        episode_rewards.append(score)
        
        # Agent end-of-episode logic (e.g. epsilon decay)
        if hasattr(agent, "end_episode"):
            agent.end_episode(ep_reward)
        
        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            print(f"Episode {episode + 1}/{episodes} - Avg Score: {sum(episode_rewards[-10:]) / min(10, len(episode_rewards)):.4f}")
    
    final_avg = sum(episode_rewards[-10:]) / min(10, len(episode_rewards)) if episode_rewards else 0
    max_score = max(episode_rewards) if episode_rewards else 0
    
    return final_avg, max_score, episode_rewards


def evaluate_agent(
    agent: Callable,
    env: PCBEnv,
    episodes: int = 10,
    model_name: str = "Agent",
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate a trained agent (no training, deterministic policy).
    
    Args:
        agent: Trained agent instance
        env: PCBEnv instance
        episodes: Number of evaluation episodes
        model_name: Name of model for logging
        verbose: Print progress
        
    Returns:
        Tuple of (average_score, std_dev)
    """
    import numpy as np
    
    logger = StructuredLogger(model_name=model_name, task_name="evaluation")
    scores = []
    
    for ep in range(episodes):
        logger.start()
        
        obs = env.reset()
        step = 0
        
        while not env.done:
            step += 1
            action_str = agent.select_action(obs, training=False)  # No exploration
            action_obj = PCBAction(action=action_str)
            next_obs, reward, done, info = env.step(action_obj)
            logger.step(step, action_str, reward.reward, done)
            obs = next_obs
        
        score = env.get_episode_score()
        scores.append(score)
        logger.end(success=True, total_steps=step, score=score, rewards=[score] * step)
        
        if verbose:
            print(f"Eval Episode {ep + 1}/{episodes} - Score: {score:.4f}")
    
    avg_score = np.mean(scores)
    std_dev = np.std(scores)
    
    return avg_score, std_dev
