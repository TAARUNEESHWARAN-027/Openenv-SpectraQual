import random

# ---------------------------
# FACTORY STATE
# ---------------------------
factory = {
    "soldering_slots": [0, 0, 0]   # each slot = remaining time
}

# ---------------------------
# PCB GENERATION
# ---------------------------
def generate_pcb():
    defects = ["none", "missing_component", "solder_bridge", "short_circuit"]

    return {
        "board_id": f"SQ-{random.randint(1000,9999)}",
        "defect_type": random.choice(defects),
        "component_cost": round(random.uniform(10, 200), 2),
        "criticality": round(random.uniform(0.1, 1.0), 2)
    }

# ---------------------------
# TIME UPDATE (REAL SIMULATION)
# ---------------------------
def update_factory():
    for i in range(len(factory["soldering_slots"])):
        if factory["soldering_slots"][i] > 0:
            factory["soldering_slots"][i] -= 1

# ---------------------------
# ASSIGN JOB TO SLOT
# ---------------------------
def assign_soldering_job():
    for i in range(len(factory["soldering_slots"])):
        if factory["soldering_slots"][i] == 0:
            factory["soldering_slots"][i] = 2  # takes 2 time units
            return True
    return False

# ---------------------------
# DECISION LOGIC
# ---------------------------
def decide_action(pcb):
    defect = pcb["defect_type"]
    cost = pcb["component_cost"]
    critical = pcb["criticality"]

    if defect == "none":
        return "PASS"

    if defect == "missing_component":
        return "ROUTE_COMPONENT_REPLACEMENT" if cost > 50 else "SCRAP"

    if defect == "solder_bridge":
        # try repair → else WAIT (smarter than scrap)
        if 0 in factory["soldering_slots"]:
            return "ROUTE_SOLDERING"
        else:
            return "WAIT"

    if defect == "short_circuit":
        return "SCRAP" if critical > 0.7 else "ROUTE_DIAGNOSTICS"

    return "SCRAP"

# ---------------------------
# REWARD FUNCTION (FINAL)
# ---------------------------
def calculate_reward(pcb, decision):
    defect = pcb["defect_type"]
    cost = pcb["component_cost"]
    critical = pcb["criticality"]

    reward = 0
    penalty = 0

    # ---------------------------
    # CASE 1: No defect
    # ---------------------------
    if defect == "none":
        if decision == "PASS":
            reward += 10
        else:
            penalty += 5

    # ---------------------------
    # CASE 2: Missing component
    # ---------------------------
    elif defect == "missing_component":
        if decision == "ROUTE_COMPONENT_REPLACEMENT":
            reward += cost * 0.8
            penalty += 5   # effort cost
        elif decision == "SCRAP":
            penalty += cost * 0.5

    # ---------------------------
    # CASE 3: Solder bridge
    # ---------------------------
    elif defect == "solder_bridge":

        if decision == "ROUTE_SOLDERING":
            assigned = assign_soldering_job()

            if assigned:
                reward += cost * 0.7      # value recovered
                penalty += 10             # time delay cost (2 units * 5)
            else:
                penalty += 50             # bottleneck

        elif decision == "WAIT":
            penalty += 10                 # waiting loss

        else:  # SCRAP
            penalty += cost * 0.6

    # ---------------------------
    # CASE 4: Short circuit
    # ---------------------------
    elif defect == "short_circuit":

        if decision == "SCRAP":
            reward += 20

        elif decision == "ROUTE_DIAGNOSTICS":
            reward += 10      # benefit
            penalty += 5      # cost

        elif decision not in ["SCRAP", "ROUTE_DIAGNOSTICS"]:
            penalty += 30 * critical   # risky wrong routing

    return reward - penalty