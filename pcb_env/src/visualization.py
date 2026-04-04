"""
Visualization - Live plotting and charts for evaluation metrics
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import json


class EpisodeVisualizer:
    """Visualize episode metrics dynamically."""
    
    def __init__(self):
        self.steps = []
        self.rewards = []
        self.actions = []
    
    def add_step(self, step_num: int, action: str, reward: float):
        """Add a step to the visualization data."""
        self.steps.append(step_num)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def plot_rewards(self, title: str = "Episode Rewards") -> str:
        """Generate reward over-time plot."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.steps,
            y=self.rewards,
            mode='lines+markers',
            name='Step Reward',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add cumulative reward line
        cumulative = []
        total = 0
        for r in self.rewards:
            total += r
            cumulative.append(total)
        
        fig.add_trace(go.Scatter(
            x=self.steps,
            y=cumulative,
            mode='lines',
            name='Cumulative Reward',
            line=dict(color='red', width=1, dash='dash'),
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title="Reward",
            hovermode='x unified',
            template="plotly_white",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def plot_action_distribution(self, title: str = "Action Distribution") -> str:
        """Generate action frequency chart."""
        action_counts = {}
        for action in self.actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(action_counts.keys()),
                y=list(action_counts.values()),
                marker=dict(color='steelblue')
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Action",
            yaxis_title="Count",
            template="plotly_white",
            height=350
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def get_summary(self) -> Dict[str, Any]:
        """Get episode summary stats."""
        return {
            "total_steps": len(self.steps),
            "total_reward": sum(self.rewards),
            "average_reward": sum(self.rewards) / len(self.rewards) if self.rewards else 0,
            "max_reward": max(self.rewards) if self.rewards else 0,
            "min_reward": min(self.rewards) if self.rewards else 0,
            "unique_actions": len(set(self.actions)),
        }


def compare_models_chart(model_scores: Dict[str, float], title: str = "Model Comparison") -> str:
    """Generate comparison chart for multiple models."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(model_scores.keys()),
            y=list(model_scores.values()),
            marker=dict(color=['green' if v > 0.7 else 'orange' for v in model_scores.values()])
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Score",
        template="plotly_white",
        height=400
    )
    
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Baseline (0.7)")
    
    return fig.to_html(include_plotlyjs='cdn')
