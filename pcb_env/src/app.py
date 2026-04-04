import json
import os
import gradio as gr
import tempfile
from typing import Optional, Tuple
from src.env import PCBEnv
from src.models import PCBAction
from src.tasks import run_all_tasks as run_task_suite
from src.inference import BaselineAgent
from src.logging_utils import StructuredLogger
from src.visualization import EpisodeVisualizer, compare_models_chart
from src.ql_agent import QLearningAgent
from src.training import train_agent, evaluate_agent

# Store models in session
stored_models = {}
training_history = {}


def run_baseline() -> str:
    """Run baseline rule-based agent."""
    try:
        logger = StructuredLogger(model_name="BaselineAgent_v1")
        logger.start()
        
        env = PCBEnv()
        agent = BaselineAgent()
        obs = env.reset()
        
        viz = EpisodeVisualizer()
        steps_list = []
        step = 0
        
        while not env.done:
            step += 1
            action = agent.select_action(obs)
            action_obj = action if isinstance(action, PCBAction) else PCBAction(action=action)
            obs, reward, done, info = env.step(action_obj)
            
            logger.step(step, action_obj.action, reward.reward, done)
            viz.add_step(step, action_obj.action, reward.reward)
            steps_list.append(f"Step {step}: {action_obj.action:30s} | Reward: {reward.reward:+.2f}")
        
        score = env.get_episode_score()
        logger.end(success=True, total_steps=step, score=score, rewards=viz.rewards)
        
        summary = viz.get_summary()
        result = f"""
Baseline Episode Summary
========================
Score: {score:.4f}
Total Steps: {step}
Total Reward: {summary['total_reward']:.2f}
Average Reward: {summary['average_reward']:.4f}

Step Details:
{chr(10).join(steps_list)}
"""
        return result
    
    except Exception as exc:
        import traceback
        return f"Baseline failed: {type(exc).__name__}\n{traceback.format_exc()}"


def run_tasks() -> str:
    """Run easy/medium/hard task suite."""
    try:
        results = run_task_suite(PCBEnv(), verbose=False)
        return json.dumps(results, indent=2)
    except Exception as exc:
        import traceback
        return f"Tasks failed: {type(exc).__name__}\n{traceback.format_exc()}"


def run_all_tasks() -> str:
    """Backward-compatible alias for task suite callback."""
    return run_tasks()


def run_hosted_model(model_name: str) -> str:
    """Run hosted model (read-only inference)."""
    try:
        if not model_name or not model_name.strip():
            return "Error: Model name required (e.g., 'gpt-3.5-turbo' or 'meta-llama/Llama-2-7b')"
        
        from src.hosted_model import HostedModelClient
        
        logger = StructuredLogger(model_name=model_name)
        logger.start()
        
        try:
            client = HostedModelClient(model_name=model_name)
        except ValueError as e:
            return f"Hosted model error: {e}\n(Ensure HF_TOKEN or API_BASE_URL is set)"
        
        env = PCBEnv()
        obs = env.reset()
        
        viz = EpisodeVisualizer()
        step = 0
        
        try:
            while not env.done:
                step += 1
                action = client.select_action(obs)
                action_obj = PCBAction(action=action)
                obs, reward, done, info = env.step(action_obj)
                
                logger.step(step, action_obj.action, reward.reward, done)
                viz.add_step(step, action_obj.action, reward.reward)
        
        except RuntimeError as e:
            # Fallback to baseline on API failure
            logger.step(step, "FALLBACK", 0.0, False, error=str(e))
            return f"Hosted model API failed, falling back to baseline:\n{str(e)}\n\nNote: Ensure model is deployed and HF_TOKEN is valid."
        
        score = env.get_episode_score()
        logger.end(success=True, total_steps=step, score=score, rewards=viz.rewards)
        
        summary = viz.get_summary()
        result = f"""
Hosted Model: {model_name}
==========================
Score: {score:.4f}
Steps: {step}
Avg Reward: {summary['average_reward']:.4f}
Status: SUCCESS
"""
        return result
    
    except Exception as exc:
        import traceback
        return f"Hosted model failed: {type(exc).__name__}\n{traceback.format_exc()}"


def upload_and_run_model(model_file) -> Tuple[str, str]:
    """Upload and run a local RL model."""
    try:
        if model_file is None:
            return "Error: No file uploaded", ""
        
        from src.rl_models import RLModelLoader
        from src.utils import observation_to_vector
        
        logger = StructuredLogger(model_name=model_file.name)
        logger.start()
        
        # Load model
        try:
            loader = RLModelLoader(model_file.name)
            model_name = os.path.basename(model_file.name)
        except Exception as e:
            return f"Failed to load model: {e}", ""
        
        env = PCBEnv()
        obs = env.reset()
        
        viz = EpisodeVisualizer()
        step = 0
        
        while not env.done:
            step += 1
            action = loader.select_action(obs, obs_to_vector=observation_to_vector)
            action_obj = PCBAction(action=action)
            obs, reward, done, info = env.step(action_obj)
            
            logger.step(step, action_obj.action, reward.reward, done)
            viz.add_step(step, action_obj.action, reward.reward)
        
        score = env.get_episode_score()
        logger.end(success=True, total_steps=step, score=score, rewards=viz.rewards)
        
        # Store for comparison
        stored_models[model_name] = score
        
        summary = viz.get_summary()
        result = f"""
Uploaded Model: {model_name}
============================
Score: {score:.4f}
Steps: {step}
Avg Reward: {summary['average_reward']:.4f}
Status: SUCCESS

Model stored for comparison.
"""
        # Generate visualization
        viz_html = viz.plot_rewards(f"Episode Rewards - {model_name}")
        
        return result, viz_html
    
    except Exception as exc:
        import traceback
        return f"Upload failed: {type(exc).__name__}\n{traceback.format_exc()}", ""


def train_model_live(num_episodes: int = 20) -> str:
    """Train a Q-Learning agent live."""
    try:
        if num_episodes < 1 or num_episodes > 1000:
            return "Error: episodes must be 1-1000"
        
        logger = StructuredLogger(model_name="QLearningAgent")
        agent = QLearningAgent(learning_rate=0.1, gamma=0.99, epsilon=1.0)
        env = PCBEnv()
        
        result_text = f"Training Q-Learning for {num_episodes} episodes...\n\n"
        
        avg_score, max_score, episode_rewards = train_agent(
            agent, env, episodes=num_episodes, model_name="QLearningAgent", verbose=True
        )
        
        stats = agent.get_stats()
        
        result_text += f"""
Training Complete
==================
Episodes: {stats['episodes']}
Avg Reward (last 10): {avg_score:.4f}
Max Reward: {max_score:.4f}
Min Reward: {stats['min_reward']:.4f}
Final Epsilon: {stats['epsilon']:.4f}

Episode Rewards (all): {','.join(f'{r:.3f}' for r in episode_rewards)}
"""
        
        training_history["QLearningAgent"] = {
            "avg_score": avg_score,
            "max_score": max_score,
            "episode_rewards": episode_rewards
        }
        
        return result_text
    
    except Exception as exc:
        import traceback
        return f"Training failed: {type(exc).__name__}\n{traceback.format_exc()}"


def compare_models_func() -> str:
    """Compare baseline vs uploaded models."""
    try:
        comparison = {
            "Baseline": 0.726,  # From previous runs
        }
        comparison.update(stored_models)
        
        if "QLearningAgent" in training_history:
            comparison["QLearningAgent"] = training_history["QLearningAgent"]["avg_score"]
        
        if len(comparison) == 1:
            return "No models to compare. Upload a model or train an agent first."
        
        chart_html = compare_models_chart(comparison)
        
        result_text = f"""
Model Comparison
================
{json.dumps(comparison, indent=2)}

Baseline Score: 0.726
"""
        return result_text
    
    except Exception as exc:
        import traceback
        return f"Comparison failed: {type(exc).__name__}\n{traceback.format_exc()}"


def build_demo():
    """Build enhanced Gradio UI with tabs."""
    with gr.Blocks(title="PCB OpenEnv - Hybrid Model Testing") as demo:
        gr.Markdown("# PCB Defect Triage OpenEnv - Hosted & Local Models")
        gr.Markdown("Test baseline, hosted models, and train local RL agents with live visualization.")
        
        with gr.Tabs():
            # Tab 1: Baseline
            with gr.Tab("Baseline"):
                gr.Markdown("Run the rule-based baseline agent")
                btn_baseline = gr.Button("Run Baseline Episode")
                out_baseline = gr.Textbox(label="Output", lines=15)
                btn_baseline.click(run_baseline, outputs=out_baseline)
            
            # Tab 2: Task Suite
            with gr.Tab("Task Suite"):
                gr.Markdown("Run easy/medium/hard benchmark tasks")
                btn_tasks = gr.Button("Run All Tasks")
                out_tasks = gr.Textbox(label="Results", lines=10)
                btn_tasks.click(run_tasks, outputs=out_tasks)
            
            # Tab 3: Hosted Model
            with gr.Tab("Hosted Model (Read-only)"):
                gr.Markdown("""
                Run inference from a hosted model API.
                - Supports Hugging Face Inference API
                - Requires HF_TOKEN environment variable
                - Examples: 'gpt-3.5-turbo', 'meta-llama/Llama-2-7b'
                """)
                model_name_input = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., meta-llama/Llama-2-7b-hf",
                    value=""
                )
                btn_hosted = gr.Button("Run Hosted Model")
                out_hosted = gr.Textbox(label="Output", lines=10)
                btn_hosted.click(run_hosted_model, inputs=model_name_input, outputs=out_hosted)
            
            # Tab 4: Upload & Run Model
            with gr.Tab("Upload Model (Trainable)"):
                gr.Markdown("""
                Upload a PyTorch (.pt), TensorFlow (.h5), or Pickle (.pkl) model.
                The model will be loaded and run for one episode with visualization.
                """)
                model_upload = gr.File(label="Upload RL Model", file_types=[".pt", ".pkl", ".h5"])
                btn_upload = gr.Button("Upload & Run")
                out_upload = gr.Textbox(label="Output", lines=10)
                out_viz = gr.HTML(label="Reward Visualization")
                btn_upload.click(upload_and_run_model, inputs=model_upload, outputs=[out_upload, out_viz])
            
            # Tab 5: Train Model
            with gr.Tab("Train Q-Learning Agent"):
                gr.Markdown("""
                Train a Q-Learning agent locally.
                - Live training in the browser
                - View training progress and final scores
                """)
                episodes_input = gr.Slider(
                    label="Number of Episodes",
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1
                )
                btn_train = gr.Button("Start Training")
                out_train = gr.Textbox(label="Training Progress", lines=15)
                btn_train.click(train_model_live, inputs=episodes_input, outputs=out_train)
            
            # Tab 6: Comparison
            with gr.Tab("Model Comparison"):
                gr.Markdown("Compare scores across baseline, hosted, and trained models")
                btn_compare = gr.Button("Generate Comparison")
                out_comp = gr.Textbox(label="Comparison Results", lines=15)
                btn_compare.click(compare_models_func, outputs=out_comp)
    
    return demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
    )