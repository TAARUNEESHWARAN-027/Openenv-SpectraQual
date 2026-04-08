"""
config.py — SpectraQual Centralized Configuration
All constants, reward weights, task definitions, and environment settings live here.
"""

# ---------------------------
# DEFECT TYPES
# ---------------------------
DEFECT_TYPES = ["none", "missing_component", "solder_bridge", "short_circuit"]

# ---------------------------
# ACTION SPACE
# ---------------------------
ACTIONS = [
    "PASS",
    "SCRAP",
    "ROUTE_COMPONENT_REPLACEMENT",
    "ROUTE_SOLDERING",
    "ROUTE_DIAGNOSTICS",
    "WAIT",
]

# Valid actions per defect type
VALID_ACTIONS = {
    "none":              ["PASS"],
    "missing_component": ["ROUTE_COMPONENT_REPLACEMENT", "SCRAP"],
    "solder_bridge":     ["ROUTE_SOLDERING", "WAIT", "SCRAP"],
    "short_circuit":     ["SCRAP", "ROUTE_DIAGNOSTICS"],
}

# ---------------------------
# FACTORY SETTINGS
# ---------------------------
N_SOLDERING_SLOTS = 3          # Number of parallel soldering slots
SOLDERING_JOB_DURATION = 2     # Time units a soldering job occupies a slot

# ---------------------------
# PCB GENERATION BOUNDS
# ---------------------------
COMPONENT_COST_MIN = 10.0
COMPONENT_COST_MAX = 200.0
CRITICALITY_MIN    = 0.1
CRITICALITY_MAX    = 1.0

# Anomaly: board_id prefix for rare-defect boards
ANOMALY_COST_THRESHOLD       = 180.0   # cost > this → anomaly candidate
ANOMALY_CRITICALITY_THRESHOLD = 0.92   # criticality > this → anomaly candidate

# ---------------------------
# REWARD WEIGHTS (multi-component)
# ---------------------------
REWARD_WEIGHT_DEFECT      = 0.35
REWARD_WEIGHT_COST        = 0.25
REWARD_WEIGHT_QUEUE       = 0.20
REWARD_WEIGHT_CRITICALITY = 0.10
REWARD_WEIGHT_ANOMALY     = 0.10

# Raw reward scaling reference (used for normalization)
RAW_REWARD_MIN = -60.0
RAW_REWARD_MAX = 160.0

# ---------------------------
# TASK DEFINITIONS
# ---------------------------
TASKS = {
    "task_easy": {
        "id":          "task_easy",
        "description": "Triage 10 boards with no slot pressure. Focus: correct defect classification.",
        "difficulty":  "easy",
        "n_boards":    10,
        "seed":        42,
        "n_slots":     3,       # all slots always available
        "anomaly_rate": 0.0,
    },
    "task_medium": {
        "id":          "task_medium",
        "description": "Triage 15 boards with one soldering slot. Manage queue pressure.",
        "difficulty":  "medium",
        "n_boards":    15,
        "seed":        99,
        "n_slots":     1,       # only 1 slot → queue pressure
        "anomaly_rate": 0.1,
    },
    "task_hard": {
        "id":          "task_hard",
        "description": "Triage 20 boards with mixed anomalies and tight slot constraints.",
        "difficulty":  "hard",
        "n_boards":    20,
        "seed":        777,
        "n_slots":     1,
        "anomaly_rate": 0.25,
    },
}

# Grader thresholds
MEDIUM_ECONOMIC_TARGET   = 0.50   # 50% of max possible economic reward
HARD_ANOMALY_RATE_TARGET = 0.50   # must flag ≥50% of actual anomalies

# ---------------------------
# INFERENCE SCRIPT SETTINGS
# ---------------------------
MAX_STEPS_PER_TASK    = 25        # safety cap (must fit in 20-min runtime)
SUCCESS_SCORE_THRESHOLD = 0.60    # ≥0.60 normalized score = success
TEMPERATURE           = 0.2
MAX_TOKENS            = 64        # actions are short, no need for long outputs

# ---------------------------
# LOGGING COLOR REFERENCE (for app.py)
# ---------------------------
COLOR_PRIMARY  = "#00e5ff"
COLOR_SUCCESS  = "#00e676"
COLOR_WARNING  = "#ffb700"
COLOR_DANGER   = "#ff3232"
COLOR_BG       = "#080c12"
COLOR_CARD     = "#0d1b2a"
COLOR_MUTED    = "#2e6a80"
