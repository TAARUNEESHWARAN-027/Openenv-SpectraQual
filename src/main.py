from env import generate_pcb, decide_action, calculate_reward
from env import update_factory, factory

TOTAL_BOARDS = 10
total_score = 0

# Reset factory
factory["soldering_slots"] = [0, 0, 0]

for i in range(TOTAL_BOARDS):

    print(f"\n--- TIME STEP {i+1} ---")

    #Update factory (time passes)
    update_factory()

    pcb = generate_pcb()
    decision = decide_action(pcb)
    reward = calculate_reward(pcb, decision)

    total_score += reward

    print(f"PCB: {pcb}")
    print(f"Decision: {decision}")
    print(f"Reward: {round(reward,2)}")
    print(f"Factory Slots: {factory['soldering_slots']}")

print("\n⚔️ Total Economic Score:", round(total_score,2))