from env import generate_pcb, calculate_reward, update_factory, factory
from agent import get_state, choose_action, update_q

EPISODES = 500
STEPS_PER_EPISODE = 20   # multi-step episodes

for ep in range(EPISODES):

    factory["soldering_slots"] = [0, 0, 0]

    pcb = generate_pcb()
    state = get_state(pcb, factory)

    for step in range(STEPS_PER_EPISODE):

        action = choose_action(state)

        update_factory()

        reward = calculate_reward(pcb, action)

        next_pcb = generate_pcb()
        next_state = get_state(next_pcb, factory)

        update_q(state, action, reward, next_state)

        # move forward
        pcb = next_pcb
        state = next_state

print("Training Complete")