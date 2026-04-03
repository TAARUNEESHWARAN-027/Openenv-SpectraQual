import random

# Q-table
Q = {}

# Actions
ACTIONS = [
    "PASS",
    "SCRAP",
    "ROUTE_COMPONENT_REPLACEMENT",
    "ROUTE_SOLDERING",
    "ROUTE_DIAGNOSTICS",
    "WAIT"
]
def get_valid_actions(defect):
    if defect == "none":
        return ["PASS"]

    if defect == "missing_component":
        return ["ROUTE_COMPONENT_REPLACEMENT", "SCRAP"]

    if defect == "solder_bridge":
        return ["ROUTE_SOLDERING", "WAIT", "SCRAP"]

    if defect == "short_circuit":
        return ["SCRAP", "ROUTE_DIAGNOSTICS"]

    return ["SCRAP"]

# Convert PCB → STATE
def get_state(pcb, factory):
    slots_free = factory["soldering_slots"].count(0)

    return (
        pcb["defect_type"],
        round(pcb["component_cost"] / 50),   # bucket cost
        round(pcb["criticality"], 1),
        slots_free
    )

# Initialize state
def init_state(state):
    if state not in Q:
        Q[state] = {a: 0 for a in ACTIONS}

# Epsilon-Greedy policy
def choose_action(state, epsilon=0.3):
    init_state(state)

    defect = state[0]
    valid_actions = get_valid_actions(defect)

    # Exploration
    if random.random() < epsilon:
        return random.choice(valid_actions)

    # Exploitation (best action among valid ones)
    return max(valid_actions, key=lambda a: Q[state][a])

# Q-learning update
def update_q(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    init_state(next_state)

    old = Q[state][action]
    future = max(Q[next_state].values())

    Q[state][action] = old + alpha * (reward + gamma * future - old)