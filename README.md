# ⚔️ SpectraQual — PCB Smart Quality-Control OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-00e5ff?style=flat-square)](https://github.com/openenv)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**SpectraQual** is a real-world AI environment that simulates smart quality-control triage for Printed Circuit Boards (PCBs) in a manufacturing factory.

An AI agent receives a stream of PCBs, each with a different defect type, component cost, and criticality score. The agent must choose the optimal economic action (Pass, Scrap, Route to Repair, Wait) while managing a shared factory soldering slot queue.

> **Why this problem matters:** PCB triage is a real, high-stakes manufacturing task. Wrong decisions mean wasted boards, bottlenecked production lines, and downstream electronics failures. SpectraQual models this as an RL environment where an agent must balance economic value, operational constraints, and risk — a setting where LLM agents can be meaningfully evaluated.

---

##  Environment Overview

| Property | Value |
|---|---|
| **Domain** | Smart Manufacturing / Industrial AI |
| **Tasks** | 3 (Easy → Hard) |
| **Action Space** | 6 discrete actions |
| **Observation Space** | 13 fields (typed Pydantic model) |
| **Reward Range** | `[0.0, 1.0]` normalized |
| **Reward Signal** | Dense (per-step), 5 components |
| **Seeded / Reproducible** | ✅ Yes |
| **Anomaly Detection** | ✅ Yes |
| **OpenEnv Spec** | ✅ Compliant |

---

##  Action Space

| Action | Description | Valid When |
|---|---|---|
| `PASS` | Clear the board — no defect | `defect_type = none` |
| `SCRAP` | Discard the board | Any defect |
| `ROUTE_COMPONENT_REPLACEMENT` | Send to component repair | `missing_component` |
| `ROUTE_SOLDERING` | Send to soldering station | `solder_bridge` |
| `ROUTE_DIAGNOSTICS` | Send for investigation | `short_circuit` |
| `WAIT` | Hold board until slot free | `solder_bridge` (no slot) |

---

##  Observation Space

```python
class PCBObservation(BaseModel):
    board_id: str                   # Unique PCB ID (e.g. "SQ-4321")
    defect_type: str                # "none" | "missing_component" | "solder_bridge" | "short_circuit"
    component_cost: float           # Replacement cost ₹10–200
    criticality: float              # Risk score 0.1–1.0
    slots_free: int                 # Available soldering slots
    slots_state: List[int]          # Remaining time per slot (0=free, -1=locked)
    is_anomaly: bool                # True if board is rare/extreme
    anomaly_score: float            # Anomaly confidence 0.0–1.0
    valid_actions: List[str]        # Permitted actions for this defect
    rolling_accuracy: float         # Fraction of correct decisions so far
    throughput: float               # Boards/step so far
    cumulative_reward: float        # Episode cumulative reward
    step: int                       # Current step number
```

---

##  Reward Function

Reward is **dense** (given every step) and **decomposed into 5 interpretable components**, all normalized to `[0.0, 1.0]`:

| Component | Weight | Description |
|---|---|---|
| `defect_reward` | 35% | Correctness of the action for the defect type |
| `cost_efficiency` | 25% | Economic value retained vs. lost |
| `queue_penalty` | 20% | Factory bottleneck avoidance |
| `criticality_factor` | 10% | Risk-adjusted multiplier |
| `anomaly_bonus` | 10% | Correct handling of anomalous boards |

**Final reward** = weighted sum of all 5 components, clamped to `[0.0, 1.0]`.

Every `StepResult` includes a full `RewardComponents` object with an `explanation` field explaining why the reward was given — enabling full explainability.

---

##  Tasks

### Task Easy (`task_easy`)
- **Boards:** 10 | **Seed:** 42 | **Slots:** 3 | **Anomaly Rate:** 0%
- **Objective:** Correctly classify all defect types. No slot pressure.
- **Grader:** `0.70 × accuracy + 0.30 × avg_reward`
- **Expected frontier model score:** ≥ 0.85

### Task Medium (`task_medium`)
- **Boards:** 15 | **Seed:** 99 | **Slots:** 1 | **Anomaly Rate:** 10%
- **Objective:** Triage boards with one soldering slot — manage queue pressure.
- **Grader:** `0.60 × economic_efficiency + 0.40 × bottleneck_avoidance`
- **Expected frontier model score:** ≥ 0.65

### Task Hard (`task_hard`)
- **Boards:** 20 | **Seed:** 777 | **Slots:** 1 | **Anomaly Rate:** 25%
- **Objective:** Handle anomalous boards safely AND maintain throughput with tight slots.
- **Grader:** `0.50 × anomaly_score + 0.30 × economic_score + 0.20 × throughput_score`
- **Expected frontier model score:** ≥ 0.50

---

##  Setup & Usage

### Prerequisites

```bash
Python >= 3.11
pip install -r requirements.txt
```

### 1) Launch the Streamlit Dashboard

```bash
streamlit run src/app.py
```

### 2) Run the LLM Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="meta-llama/llama-3.3-70b-instruct"
export HF_TOKEN="hf_your_key_here"

# Run baseline inference
python inference.py
```

### 3) Run Task Grader Sanity Check

```bash
cd src
python tasks.py
```

### 4) Train the Q-learning Agent

```bash
python src/train.py
```

### 5) Run CLI Simulation (rule-based)

```bash
python src/main.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t spectraqual .

# Run the API server (default — what HF Spaces runs)
# Exposes: GET / | POST /reset | POST /step | GET /state
docker run -p 7860:7860 spectraqual
# → API docs available at http://localhost:7860/docs

# Run inference inside container
docker run \
  -e API_BASE_URL="https://openrouter.ai/api/v1" \
  -e MODEL_NAME="meta-llama/llama-3.3-70b-instruct" \
  -e HF_TOKEN="hf_..." \
  --entrypoint python spectraqual inference.py

# Run Streamlit dashboard locally (NOT the Docker default — local dev only)
streamlit run src/app.py --server.port 8501
```

---

## 📁 Project Structure

```
spectraqual/
├── inference.py          # Root LLM baseline script (required by OpenEnv)
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # Container definition
├── requirements.txt      # Pinned dependencies
├── README.md             # This file
└── src/
    ├── config.py         # All constants, task configs, reward weights
    ├── models.py         # Pydantic typed models (Observation, Action, Reward)
    ├── env.py            # SpectraQualEnv class (reset/step/state + legacy wrappers)
    ├── reward.py         # Multi-component normalized reward calculator
    ├── tasks.py          # 3 tasks + programmatic graders
    ├── agent.py          # Q-learning agent (baseline model zoo)
    ├── app.py            # Streamlit dashboard
    ├── train.py          # Offline Q-table trainer
    └── main.py           # Rule-based CLI runner
```

---

## 📊 Baseline Scores

| Agent | task_easy | task_medium | task_hard |
|---|---|---|---|
| Rule-based | ~0.82 | ~0.61 | ~0.48 |
| LLM (llama-3.3-70b) | TBD | TBD | TBD |
| Q-learning (trained) | TBD | TBD | TBD |

---

## 🔬 Research Extensions

The environment supports:
- **Anomaly detection mode**: boards with extreme cost+criticality are flagged
- **Seeded reproducibility**: every task uses a fixed RNG seed
- **Pluggable agents**: any agent implementing `predict(observation) → action`
- **Dense reward signal**: sub-rewards for debugging and ablation studies
- **Explainability**: every step reward comes with a natural-language explanation
- **Benchmark modes**: noisy observations, partial observability (planned)

---

## ⚙️ Environment Variables for Inference

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `https://openrouter.ai/api/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `meta-llama/llama-3.3-70b-instruct` | Model identifier |
| `HF_TOKEN` | Yes (prod) | — | Hugging Face / API key |
| `LOCAL_IMAGE_NAME` | No | — | Docker image (for containerized env) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
