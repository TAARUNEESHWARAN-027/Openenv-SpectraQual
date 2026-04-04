# SpectraQual

SpectraQual is a simulated smart quality-control decision system for PCB (Printed Circuit Board) triage. It models incoming boards with different defect types, chooses an action (pass, repair route, wait, or scrap), and scores each decision using an economic reward function.

The project combines:
- A rule-informed environment model of defects, costs, criticality, and factory slot capacity.
- A Q-learning agent that learns action values over abstracted state buckets.
- A Streamlit dashboard for real-time simulation, visibility, and decision analytics.

## Purpose

The current implementation is designed to demonstrate how AI-assisted decisioning can optimize manufacturing outcomes under operational constraints.

Core goals:
- Maximize economic score across a stream of boards.
- Balance repair value against risk, delay, and resource bottlenecks.
- Visualize decisions and factory capacity in real time.

## How The System Works

Each simulation step:
1. Factory time advances (occupied soldering slots count down).
2. A random PCB is generated with:
	 - defect type
	 - component replacement cost
	 - criticality score
3. The current board + factory context are converted into an RL state.
4. The agent chooses a valid action for that defect.
5. Reward is computed from defect/action economics.
6. Score and history are updated for trend analysis.

## Project Structure (Current Behavior)

- `src/app.py`
	- Streamlit UI application.
	- Runs one-step or auto-run simulation loops.
	- Displays PCB details, chosen action, confidence indicator, reward, slot occupancy, cumulative score, trend plot, and decision distribution.
	- Uses the RL agent for decision selection (`epsilon=0` in UI, i.e., greedy policy).

- `src/env.py`
	- Environment and business logic source of truth.
	- Maintains global factory state (`soldering_slots`).
	- Generates random PCBs.
	- Updates slot timers.
	- Assigns soldering jobs when capacity is available.
	- Contains baseline rule policy (`decide_action`) and an environment reward function.

- `src/agent.py`
	- Q-learning implementation.
	- Defines action space and valid action constraints per defect type.
	- Converts raw inputs into a compact state tuple.
	- Implements epsilon-greedy action selection and Q-value updates.

- `src/reward.py`
	- Separate reward function module used by the Streamlit app.
	- Mirrors the same reward intent as `env.py` and references factory slot availability.
	- Note: this duplication means reward logic exists in two places.

- `src/train.py`
	- Offline trainer for Q-table learning.
	- Runs multiple episodes and steps, updates Q-values via temporal-difference learning.

- `src/main.py`
	- Terminal simulation runner using rule-based decision logic from `env.py`.
	- Useful for quick sanity checks without UI.

- `src/config.py`
	- Currently empty (placeholder for future centralized configuration).

- `src/models.py`
	- Currently empty (placeholder for typed data models/domain classes).

- `requirements.txt`
	- Currently empty.
	- Based on imports, the project needs at least `streamlit` and `matplotlib`.

## Running The Project

From project root:

```bash
# Optional: create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages (requirements.txt is currently empty)
pip install streamlit matplotlib
```

### 1) Launch the Streamlit dashboard

```bash
streamlit run src/app.py
```

### 2) Train the Q-learning agent

```bash
python src/train.py
```

### 3) Run CLI simulation (rule-based)

```bash
python src/main.py
```

## Decision Space

Possible actions:
- `PASS`
- `SCRAP`
- `ROUTE_COMPONENT_REPLACEMENT`
- `ROUTE_SOLDERING`
- `ROUTE_DIAGNOSTICS`
- `WAIT`

Action validity depends on defect class (for example, `none` only allows `PASS`).

## Notes On Current State

- Reward logic is duplicated in both `src/env.py` and `src/reward.py`; this may diverge over time.
- The UI currently computes a heuristic confidence value from board criticality rather than model uncertainty.
- Training and UI share the same in-memory Q-table module (`agent.py`) but no persistence is implemented yet.

## Suggested Next Improvements

1. Consolidate reward logic into one module.
2. Add Q-table save/load for reusable trained policies.
3. Populate `requirements.txt` and pin package versions.
4. Add tests for reward correctness and valid-action constraints.
