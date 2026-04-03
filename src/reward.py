from env import factory, assign_soldering_job

def calculate_reward(pcb, decision):
    defect = pcb["defect_type"]
    cost = pcb["component_cost"]
    critical = pcb.get("criticality", 0.5)

    reward = 0
    penalty = 0

    if defect == "none":
        reward = 10 if decision == "PASS" else -5

    elif defect == "missing_component":
        if decision == "ROUTE_COMPONENT_REPLACEMENT":
            reward = cost*0.8 - 5
        else:  # SCRAP
            reward = - cost*0.5

    elif defect == "solder_bridge":
        if decision == "ROUTE_SOLDERING":
            if any(s==0 for s in factory["soldering_slots"]):
                reward = cost*0.7 - 10
            else:
                reward = -50
        elif decision == "WAIT":
            reward = -10
        else:  # SCRAP
            reward = - cost*0.6

    elif defect == "short_circuit":
        if decision == "SCRAP":
            reward = 20
        elif decision == "ROUTE_DIAGNOSTICS":
            reward = 10 - 5
        else:
            reward = -30*critical

    return reward