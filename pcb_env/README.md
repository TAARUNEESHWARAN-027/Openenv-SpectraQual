---
title: PCB Defect Triage
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# PCB Defect Triage OpenEnv

Unified and current documentation for environment design, hosted deployment, hybrid model testing, training flow, visualization, and project progress.

## 1. Project Goal

Build a contest-ready OpenEnv benchmark for PCB defect triage that supports:
- Baseline rule-based evaluation
- Hosted read-only model inference (API based)
- Uploaded local model inference (trainable workflow)
- Live reward/action visualization
- Structured [START], [STEP], [END] logs for reproducible scoring

## 2. Current Status

Overall status: Production-ready for Hugging Face Space deployment and interactive testing.

Completed:
- Core OpenEnv environment and typed models
- Easy/Medium/Hard tasks and graders
- Baseline inference agent
- Hugging Face startup hardening (stable entrypoint and port binding)
- Hybrid model testing pipeline (hosted + uploaded + trainable Q-learning)
- Structured logging utilities
- Visualization module
- Enhanced Gradio UI with 6 tabs

## 3. Folder Structure

```text
pcb_env/
├── app.py
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── .env.example
├── README.md
└── src/
   ├── __init__.py
   ├── app.py
   ├── env.py
   ├── hosted_model.py
   ├── inference.py
   ├── logging_utils.py
   ├── models.py
   ├── ql_agent.py
   ├── rl_models.py
   ├── tasks.py
   ├── training.py
   ├── utils.py
   └── visualization.py
```

## 4. Core Environment

Environment class: `PCBEnv`

Observation model: `PCBObservation`
- defect_type
- criticality_score
- component_cost
- inspection_confidence
- queue_length
- available_slots

Action model: `PCBAction`
- PASS
- SCRAP
- ROUTE_COMPONENT_REPLACEMENT
- ROUTE_SOLDERING
- ROUTE_DIAGNOSTICS
- WAIT

Reward model: `PCBReward`
- scalar float reward

Task set:
- easy
- medium
- hard

## 5. Logging Compliance

All key flows use structured logs compatible with your requested format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

## 6. Hybrid Model Testing Features

### 6.1 Hosted Model Integration (Read-only)

Module: `src/hosted_model.py`

Supports:
- Hugging Face Inference API
- OpenAI-compatible/custom endpoints

Config via environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Behavior:
- No local training
- API-only inference
- Fallback/error messaging when endpoint fails

### 6.2 Uploaded Model Integration (Trainable)

Module: `src/rl_models.py`

Supported upload formats:
- `.pt` (PyTorch)
- `.pkl` / `.pickle`
- `.h5` / `.hdf5` (TensorFlow)

Behavior:
- Load model in memory
- Run local inference on environment observations
- Optional checkpoint save helper

### 6.3 Local RL Training

Modules:
- `src/ql_agent.py`
- `src/training.py`

Includes:
- Lightweight Q-learning agent
- Epsilon-greedy training loop
- Evaluation routine for repeated episode scoring

## 7. Visualization

Module: `src/visualization.py`

Includes:
- Reward-over-time chart
- Action distribution chart
- Model comparison chart (baseline vs hosted vs uploaded/trained)

## 8. Gradio UI Controls

Module: `src/app.py`

Tabs:
1. Baseline
2. Task Suite
3. Hosted Model (Read-only)
4. Upload Model (Trainable)
5. Train Q-Learning Agent
6. Model Comparison

Each run surfaces score and step-level behavior in UI text outputs.

## 9. Hugging Face Space Deployment

This README frontmatter is configured for Spaces:
- sdk: gradio
- app_file: app.py
- python_version: 3.10

Root entrypoint `app.py` uses:
- `PORT` environment variable
- `0.0.0.0` binding
- startup traceback printing for easier debugging

### 9.1 Push Steps

```bash
cd pcb_env
git add .
git commit -m "Update unified docs and hybrid OpenEnv workflow"
git push
```

If pushing to a new Space repo:

```bash
git remote add origin https://huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>
git branch -M main
git push -u origin main
```

## 10. Local Run and Smoke Tests

### 10.1 Environment Setup

```bash
cd pcb_env
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### 10.2 Callback Smoke Test

```bash
./.venv/bin/python -c "from src.app import run_baseline, run_tasks; print('imports_ok'); print(run_tasks()[:120]); out = run_baseline(); print('baseline_ok' if 'Baseline Episode Summary' in out else out[:120])"
```

### 10.3 Launch UI

```bash
./.venv/bin/python app.py
```

## 11. Runtime Configuration

Use `.env.example` as template:

```bash
API_BASE_URL=https://api-inference.huggingface.co
MODEL_NAME=<hosted_model_id>
HF_TOKEN=<your_hf_token>
PORT=7860
DEBUG=false
```

On Hugging Face, set secrets in Space Settings for tokens.

## 12. Dependencies

Current required dependencies in `requirements.txt`:
- pydantic
- numpy
- gradio
- requests
- plotly
- python-dotenv

Optional heavyweight ML dependencies are kept commented:
- torch
- tensorflow
- stable-baselines3
- gym

## 13. Error Handling Behavior

- Hosted API failure: return clear hosted-model error in UI (safe fallback path)
- Invalid uploaded model: graceful error, no app crash
- Missing token/config: explicit message in output
- Step/action failures: captured and logged with structured format

## 14. Known Notes

- Baseline score varies run-to-run because environment is stochastic.
- Import smoke tests should be run inside project venv to avoid missing dependency errors.
- The previously seen `run_tasks` import issue is fixed.

## 15. Quick Verification Checklist

- [ ] Space build succeeds
- [ ] App starts without restart loop
- [ ] Baseline tab completes full episode
- [ ] Task Suite tab returns JSON scores
- [ ] Hosted tab handles token/model errors gracefully
- [ ] Upload tab reports invalid-model errors cleanly
- [ ] Q-learning tab finishes training loop
- [ ] Comparison tab returns model score summary

## 16. Next Optional Enhancements

1. Add PPO integration for stronger RL performance
2. Persist uploaded/trained models to `/models` directory
3. Add seeded run controls in UI for strict reproducibility
4. Add real-time streaming plots during training (per-episode live update)
5. Add API rate-limit handling/backoff for hosted models
- Batch size and max slots
- Observation/action definitions
- Task baseline scores
- Performance targets

## 📝 Contest Compliance

✅ **OpenEnv-Compliant:**
- Pydantic-typed models
- `step()`, `reset()`, `state()` API
- Structured reward shaping
- Easy → Medium → Hard task progression
- Docker containerization
- YAML specification

✅ **Performance Targets:**
- Easy task score: **1.0** (perfect)
- Medium task score: **0.80** (routing decisions)
- Hard task score: **0.75+** (optimization)
- Average score target: **0.85+**

## 🎓 Extension Ideas

1. **Multi-agent**: Multiple agents managing different repair routes
2. **Curriculum Learning**: Progressive task difficulty
3. **Transfer Learning**: Pre-train on diagnostics, fine-tune on specific defect types
4. **Constrained Optimization**: Minimize cost while maximizing throughput
5. **Real-world Data**: Integrate actual PCB defect statistics

## 📞 Support

For questions or issues:
- Check OpenEnv specification: `openenv.yaml`
- Review environment code: `src/env.py`
- Run tests: `python -m pytest` (if tests added)

---


# live url :

[[https://huggingface.co/spaces/venkie07/PCB]]


**Status**: ✅ Contest-Ready | **Version**: 0.1.0 | **License**: MIT
