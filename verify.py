import sys
sys.path.insert(0, 'src')

from config import TASKS, ACTIONS, VALID_ACTIONS
from models import PCBObservation, PCBAction, RewardComponents, StepResult
from reward import calculate_reward, detect_anomaly
from env import SpectraQualEnv

print("--- Module imports: OK ---")

# Test reset and step
env = SpectraQualEnv("task_easy")
r = env.reset()
print(f"reset() -> defect={r.observation.defect_type}, step={r.observation.step}, done={r.done}")

action = r.observation.valid_actions[0]
r2 = env.step(PCBAction(action=action))
print(f"step({action}) -> reward={r2.reward:.4f}, done={r2.done}")
print(f"  expl: {r2.reward_components.explanation[:80]}")

state = env.state()
print(f"state() -> step={state['step']}, accuracy={state['rolling_accuracy']}")

# Test all 3 tasks
for tid in ["task_easy", "task_medium", "task_hard"]:
    e = SpectraQualEnv(task_id=tid)
    rr = e.reset()
    steps = 0
    while not rr.done and steps < 30:
        action_str = rr.observation.valid_actions[0]
        rr = e.step(PCBAction(action=action_str))
        steps += 1
    s = e.state()
    print(f"[{tid}] steps={steps}, cum_reward={s['cumulative_reward']:.4f}, accuracy={s['rolling_accuracy']:.2%}")

print("--- All tests: PASS ---")
