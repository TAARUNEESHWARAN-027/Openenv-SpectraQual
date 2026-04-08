"""
app.py — SpectraQual Streamlit Dashboard (v3.0)
Updated to use the new SpectraQualEnv class with OpenEnv interface.
Features:
  - Real-time stacked reward component charts
  - Per-step accuracy / throughput display
  - Action confidence from reward components
  - Anomaly flag indicators
  - Explainability: "Why this decision?"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import matplotlib.pyplot as plt
import time

from env    import SpectraQualEnv
from models import PCBAction
from config import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING,
    COLOR_DANGER,  COLOR_BG,     COLOR_CARD, COLOR_MUTED,
    TASKS,
)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="SpectraQual",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# GLOBAL STYLES
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;600;700&family=Exo+2:wght@300;400;600;800&display=swap');

.stApp {
    background-color: #080c12;
    color: #c9d4e0;
    font-family: 'Exo 2', sans-serif;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(0deg, rgba(0,0,0,0.025) 0px, rgba(0,0,0,0.025) 1px, transparent 1px, transparent 4px);
    pointer-events: none;
    z-index: 9999;
}
h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.4rem !important;
    letter-spacing: 0.12em !important;
    color: #00e5ff !important;
    text-shadow: 0 0 18px rgba(0,229,255,0.45), 0 0 40px rgba(0,229,255,0.12);
    border-bottom: 1px solid rgba(0,229,255,0.15);
    padding-bottom: 0.4rem;
}
h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    color: #2e6a80 !important;
    text-transform: uppercase;
    margin-top: 1.4rem !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1b2a, #09141f);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 10px;
    padding: 16px 20px !important;
    box-shadow: 0 0 22px rgba(0,229,255,0.05), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: border-color 0.25s;
}
[data-testid="metric-container"]:hover { border-color: rgba(0,229,255,0.38); }
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #2e6a80 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #00e5ff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0d2137, #091824);
    color: #00e5ff;
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 6px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.1em;
    font-size: 0.85rem;
    padding: 9px 18px;
    text-transform: uppercase;
    transition: all 0.2s;
    box-shadow: 0 0 10px rgba(0,229,255,0.06);
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #123450, #0d2538);
    border-color: #00e5ff;
    box-shadow: 0 0 18px rgba(0,229,255,0.22);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }
.stSuccess, .stWarning, .stInfo, .stError {
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    border-left-width: 4px !important;
}
.stSuccess { background: rgba(0,230,118,0.07)  !important; border-color: #00e676 !important; }
.stWarning { background: rgba(255,183,0,0.07)   !important; border-color: #ffb700 !important; }
.stInfo    { background: rgba(0,229,255,0.06)   !important; border-color: #00e5ff !important; }
.stError   { background: rgba(255,50,50,0.07)   !important; border-color: #ff3232 !important; }
.pcb-card {
    background: linear-gradient(135deg, #0d1b2a, #09141f);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 10px;
    padding: 18px 22px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    line-height: 2.1;
    box-shadow: inset 0 0 24px rgba(0,0,0,0.25);
}
.lbl { color: #2e6a80; font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; }
.val { color: #c9f0ff; font-weight: 600; }
.defect-badge {
    display: inline-block;
    padding: 1px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.b-none    { background: rgba(0,230,118,0.12);  color: #00e676; border: 1px solid #00e676; }
.b-missing { background: rgba(255,183,0,0.12);  color: #ffb700; border: 1px solid #ffb700; }
.b-solder  { background: rgba(255,120,0,0.12);  color: #ff7800; border: 1px solid #ff7800; }
.b-short   { background: rgba(255,50,50,0.12);  color: #ff3232; border: 1px solid #ff3232; }
.anomaly-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    background: rgba(255,0,200,0.12);
    color: #ff00c8;
    border: 1px solid #ff00c8;
    letter-spacing: 0.1em;
    animation: anomalyPulse 1.2s ease-in-out infinite;
}
@keyframes anomalyPulse {
    0%   { box-shadow: 0 0 4px rgba(255,0,200,0.2); }
    50%  { box-shadow: 0 0 16px rgba(255,0,200,0.6); }
    100% { box-shadow: 0 0 4px rgba(255,0,200,0.2); }
}
.slot-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.slot-item {
    display: flex; align-items: center; gap: 8px;
    background: #0a1420; border-radius: 6px;
    padding: 7px 13px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    border: 1px solid rgba(255,255,255,0.05);
    min-width: 128px;
}
.dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.dot-free { background:#00e676; box-shadow:0 0 7px #00e676; }
.dot-busy { background:#ff3232; box-shadow:0 0 7px #ff3232; }
.dot-lock { background:#3a3a3a; }
.free { color:#00e676; }
.busy { color:#ff5a5a; }
.lock { color:#3a3a3a; }
.rpill {
    display: inline-block;
    padding: 5px 20px;
    border-radius: 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.rpos  { background:rgba(0,230,118,0.11); color:#00e676; border:1px solid rgba(0,230,118,0.35); }
.rneg  { background:rgba(255,50,50,0.11);  color:#ff5a5a; border:1px solid rgba(255,50,50,0.35); }
.rzero { background:rgba(140,140,140,0.09);color:#888;    border:1px solid rgba(140,140,140,0.25); }
.score-big {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-shadow: 0 0 14px currentColor;
}
hr { border:none; border-top:1px solid rgba(0,229,255,0.08) !important; margin:1.2rem 0 !important; }
.idle {
    text-align:center; padding:44px 20px;
    border:1px dashed rgba(0,229,255,0.15); border-radius:12px;
    color:#1e4a5a; font-family:'Share Tech Mono',monospace;
    font-size:0.8rem; letter-spacing:0.12em; margin-top:36px;
}
.reward-row {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.74rem;
    margin-bottom: 6px;
}
.reward-label { color: #2e6a80; width: 160px; flex-shrink: 0; }
.reward-bar-wrap { flex: 1; background: #0a1420; border-radius: 4px; height: 8px; }
.reward-bar { height: 8px; border-radius: 4px; }
.reward-val { color: #c9f0ff; width: 48px; text-align: right; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0d5e70, #00e5ff) !important;
    border-radius: 4px;
}
[data-testid="stProgressBar"] {
    background: #0a1420 !important;
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 4px;
}
.stCaption {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #2e6a80 !important;
    letter-spacing: 0.1em;
}
@keyframes pulseGlow {
    0%   { box-shadow: 0 0 5px rgba(0,229,255,0.15); }
    50%  { box-shadow: 0 0 22px rgba(0,229,255,0.45); }
    100% { box-shadow: 0 0 5px rgba(0,229,255,0.15); }
}
.stSuccess, .stWarning, .stError, .stInfo {
    animation: pulseGlow 1.5s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SESSION STATE
# ---------------------------
def _init_state():
    if "env" not in st.session_state:
        st.session_state.env = None
    if "score" not in st.session_state:
        st.session_state.score = 0.0
    if "history" not in st.session_state:
        st.session_state.history = []         # cumulative reward over time
    if "running" not in st.session_state:
        st.session_state.running = False
    if "log" not in st.session_state:
        st.session_state.log = []             # list of (pcb_obs, action, rc)
    if "task_id" not in st.session_state:
        st.session_state.task_id = "task_easy"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "episode_done" not in st.session_state:
        st.session_state.episode_done = False

_init_state()

# ---------------------------
# HELPERS
# ---------------------------
def defect_badge(d):
    m = {
        "none":              ("b-none",    "✓ NONE"),
        "missing_component": ("b-missing", "⚠ MISSING COMPONENT"),
        "solder_bridge":     ("b-solder",  "⚡ SOLDER BRIDGE"),
        "short_circuit":     ("b-short",   "✗ SHORT CIRCUIT"),
    }
    cls, label = m.get(d, ("b-none", d.upper()))
    return f'<span class="defect-badge {cls}">{label}</span>'


def reward_bar_html(label, score, color="#00e5ff"):
    pct = int(score * 100)
    return (
        f'<div class="reward-row">'
        f'  <span class="reward-label">{label}</span>'
        f'  <div class="reward-bar-wrap">'
        f'    <div class="reward-bar" style="width:{pct}%;background:{color};"></div>'
        f'  </div>'
        f'  <span class="reward-val">{score:.2f}</span>'
        f'</div>'
    )


def get_env() -> SpectraQualEnv:
    if st.session_state.env is None:
        st.session_state.env = SpectraQualEnv(task_id=st.session_state.task_id)
    return st.session_state.env


# ---------------------------
# HEADER
# ---------------------------
st.title("⚔️ SPECTRAQUAL — SMART PCB DECISION SYSTEM")
st.markdown(
    '<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;'
    'color:#1e4a5a;letter-spacing:0.16em;margin-top:-10px;margin-bottom:4px;">'
    'REAL-TIME QUALITY INTELLIGENCE ENGINE · v3.0 · OpenEnv Compliant</p>',
    unsafe_allow_html=True,
)

# ---------------------------
# SIDEBAR TASK SELECTOR
# ---------------------------
with st.sidebar:
    st.markdown("### 🎯 Task Selection")
    task_choice = st.selectbox(
        "Select Task",
        options=list(TASKS.keys()),
        format_func=lambda t: f"{t} ({TASKS[t]['difficulty'].upper()})",
        index=list(TASKS.keys()).index(st.session_state.task_id),
    )
    if task_choice != st.session_state.task_id:
        st.session_state.task_id   = task_choice
        st.session_state.env       = None
        st.session_state.score     = 0.0
        st.session_state.history   = []
        st.session_state.log       = []
        st.session_state.last_result = None
        st.session_state.episode_done = False

    cfg = TASKS[st.session_state.task_id]
    st.markdown(f"""
    **Boards:** {cfg['n_boards']}  
    **Slots:** {cfg['n_slots']}  
    **Seed:** {cfg['seed']}  
    **Anomaly Rate:** {cfg['anomaly_rate']:.0%}  
    **Difficulty:** {cfg['difficulty'].upper()}
    """)
    st.markdown("---")
    speed = st.slider("⚡ Speed (s/step)", 0.2, 2.0, 0.8, step=0.1)

# ---------------------------
# SPEED (fallback if sidebar collapsed)
# ---------------------------
if "speed" not in dir():
    speed = 0.8

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# METRICS BAR
# ---------------------------
env_obj = get_env()
state   = env_obj.state()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("💰 Cumul. Reward",  f"{state['cumulative_reward']:.3f}")
m2.metric("🎯 Accuracy",       f"{state['rolling_accuracy']:.1%}")
m3.metric("⚙️ Active Slots",   sum(1 for s in state['slots'] if 0 < s < 9999))
m4.metric("🧠 Decisions",      state['total_count'])
m5.metric("⚠️ Bottlenecks",    state['bottleneck_count'])

last_r = round(st.session_state.log[-1][2].normalized, 3) if st.session_state.log else "N/A"
status_color = "#00e5ff" if st.session_state.log else "#1e4a5a"
st.markdown(f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
    color:{status_color};padding:6px 14px;border:1px solid rgba(0,229,255,0.2);
    border-radius:6px;display:inline-block;margin-top:10px;margin-bottom:4px;
    background:rgba(0,229,255,0.03);letter-spacing:0.1em;">
🟢 TASK: {st.session_state.task_id.upper()} &nbsp;·&nbsp; LAST REWARD: {last_r} &nbsp;·&nbsp; STEPS: {state['step']}
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# CONTROL BUTTONS
# ---------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("▶  RUN STEP"):
        st.session_state.running  = False
        st.session_state.run_once = True
with c2:
    if st.button("⚡  AUTO RUN"):
        st.session_state.running = True
with c3:
    if st.button("⛔  STOP"):
        st.session_state.running = False
with c4:
    if st.button("🔄  RESET"):
        env_obj.reset()
        st.session_state.score    = 0.0
        st.session_state.history  = []
        st.session_state.log      = []
        st.session_state.last_result = None
        st.session_state.episode_done = False
with c5:
    if st.button("🆕  NEW TASK"):
        st.session_state.env = None
        st.session_state.score   = 0.0
        st.session_state.history = []
        st.session_state.log     = []
        st.session_state.last_result = None
        st.session_state.episode_done = False

# ---------------------------
# CORE STEP
# ---------------------------
def run_step():
    env  = get_env()

    # Initialize if needed
    if env._done or env._current_pcb is None:
        result = env.reset()
        if result.done:
            st.session_state.episode_done = True
            return None

    # Get current obs to determine action
    obs    = env._build_observation(*__import__("reward").detect_anomaly(env._current_pcb))

    # Use rule-based decision (greedy heuristic)
    from env import decide_action
    pcb_dict = {
        "defect_type":    obs.defect_type,
        "component_cost": obs.component_cost,
        "criticality":    obs.criticality,
    }
    action_str = decide_action(pcb_dict)

    result = env.step(PCBAction(action=action_str))
    rc     = result.reward_components

    st.session_state.score     = env.state()["cumulative_reward"]
    st.session_state.history.append(st.session_state.score)
    st.session_state.log.append((result.observation, action_str, rc))
    st.session_state.last_result = result

    if result.done:
        st.session_state.episode_done = True

    return result


# ---------------------------
# DISPLAY
# ---------------------------
def display(result):
    from collections import Counter

    obs = result.observation
    rc  = result.reward_components
    col1, col2 = st.columns(2, gap="large")

    # ── LEFT ──
    with col1:
        st.subheader("PCB Info")
        anomaly_html = ""
        if obs.is_anomaly:
            anomaly_html = f'<span class="anomaly-badge">⚠️ ANOMALY {obs.anomaly_score:.2f}</span>'

        st.markdown(f"""
        <div class="pcb-card">
            <div><span class="lbl">Board ID &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                 <span class="val">{obs.board_id}</span></div>
            <div><span class="lbl">Defect Type &nbsp;&nbsp;</span>
                 {defect_badge(obs.defect_type)}</div>
            <div><span class="lbl">Component Cost </span>
                 <span class="val">₹{obs.component_cost:.2f}</span></div>
            <div><span class="lbl">Criticality &nbsp;&nbsp;&nbsp;</span>
                 <span class="val">{obs.criticality:.2f}</span></div>
            <div><span class="lbl">Anomaly &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                 {anomaly_html if anomaly_html else '<span class="val" style="color:#2e6a80;">Normal</span>'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Decision")
        action = st.session_state.log[-1][1] if st.session_state.log else "N/A"
        if action == "PASS":
            st.success(f"✅  {action}")
        elif "ROUTE" in action:
            st.warning(f"🛠️  {action}")
        elif action == "WAIT":
            st.warning("⏳  WAITING FOR SLOT AVAILABILITY")
        else:
            st.error(f"❌  {action}")

        if rc:
            st.subheader("🧠 Why this decision?")
            explanation_parts = rc.explanation.split(" | ")
            for part in explanation_parts[:3]:
                st.info(part)

        st.subheader("Step Reward")
        r = result.reward
        if r >= 0.6:
            st.markdown(f'<span class="rpill rpos">▲ {r:.4f}</span>', unsafe_allow_html=True)
        elif r >= 0.35:
            st.markdown(f'<span class="rpill rzero">● {r:.4f}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="rpill rneg">▼ {r:.4f}</span>', unsafe_allow_html=True)

        if rc:
            st.subheader("📊 Reward Component Breakdown")
            components = [
                ("Defect Handling",  rc.defect_reward,       "#00e5ff"),
                ("Cost Efficiency",  rc.cost_efficiency,     "#00e676"),
                ("Queue Mgmt",       rc.queue_penalty,       "#ffb700"),
                ("Risk Factor",      rc.criticality_factor,  "#ff7800"),
                ("Anomaly Bonus",    rc.anomaly_bonus,       "#ff00c8"),
            ]
            bars_html = ""
            for label, val, color in components:
                bars_html += reward_bar_html(label, val, color)
            st.markdown(bars_html, unsafe_allow_html=True)

        st.subheader("Rolling Metrics")
        sub1, sub2 = st.columns(2)
        with sub1:
            st.metric("🎯 Accuracy", f"{obs.rolling_accuracy:.1%}")
        with sub2:
            st.metric("⚡ Throughput", f"{obs.throughput:.2f}")

    # ── RIGHT ──
    with col2:
        st.subheader("Factory Slots")
        slot_html = '<div class="slot-grid">'
        for i, slot in enumerate(obs.slots_state):
            if slot == -1:
                slot_html += (f'<div class="slot-item"><div class="dot dot-lock"></div>'
                              f'<span class="lock">SLOT {i:02d} · LOCKED</span></div>')
            elif slot > 0:
                slot_html += (f'<div class="slot-item"><div class="dot dot-busy"></div>'
                              f'<span class="busy">SLOT {i:02d} · {slot}t</span></div>')
            else:
                slot_html += (f'<div class="slot-item"><div class="dot dot-free"></div>'
                              f'<span class="free">SLOT {i:02d} · FREE</span></div>')
        slot_html += '</div>'
        st.markdown(slot_html, unsafe_allow_html=True)

        st.subheader("Cumulative Reward")
        score_color = "#00e676" if st.session_state.score >= 0.5 else "#ff5a5a"
        st.markdown(
            f'<div class="score-big" style="color:{score_color}">'
            f'{st.session_state.score:.4f}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("📈 Reward Trend")
        fig, ax = plt.subplots(figsize=(5.5, 3))
        fig.patch.set_facecolor("#080c12")
        ax.set_facecolor("#0a1420")
        history = st.session_state.history
        if history:
            ax.plot(history, color="#00e5ff", linewidth=1.8,
                    marker='o', markersize=3.5,
                    markerfacecolor="#00e5ff", markeredgewidth=0)
            ax.fill_between(range(len(history)), history, alpha=0.10, color="#00e5ff")
            ax.axhline(y=0.6, color="#00e676", linewidth=0.8, linestyle="--", alpha=0.5, label="Success threshold")
        ax.set_title("Cumulative Reward", color="#2e6a80", fontsize=9, pad=8)
        ax.set_xlabel("Steps",  color="#2e6a80", fontsize=8)
        ax.set_ylabel("Score",  color="#2e6a80", fontsize=8)
        ax.set_ylim(0, max(max(history, default=1.0) * 1.1, 1.0))
        ax.tick_params(colors="#2e6a80", labelsize=7)
        ax.grid(color="#0d2535", linewidth=0.7, linestyle="--")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0d2535")
        fig.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close(fig)

        # Stacked Reward Components Over Time
        if len(st.session_state.log) >= 2:
            st.subheader("📊 Component Breakdown Over Time")
            steps_data = st.session_state.log[-20:]  # last 20 steps
            comp_labels = ["Defect", "Cost", "Queue", "Risk", "Anomaly"]
            comp_colors = ["#00e5ff", "#00e676", "#ffb700", "#ff7800", "#ff00c8"]
            comp_data   = {l: [] for l in comp_labels}

            for _, _, rc_entry in steps_data:
                if rc_entry:
                    comp_data["Defect"].append(rc_entry.defect_reward)
                    comp_data["Cost"].append(rc_entry.cost_efficiency)
                    comp_data["Queue"].append(rc_entry.queue_penalty)
                    comp_data["Risk"].append(rc_entry.criticality_factor)
                    comp_data["Anomaly"].append(rc_entry.anomaly_bonus)

            if any(comp_data.values()):
                fig2, ax2 = plt.subplots(figsize=(5.5, 2.8))
                fig2.patch.set_facecolor("#080c12")
                ax2.set_facecolor("#0a1420")
                x = list(range(len(next(iter(comp_data.values())))))
                bottom = [0.0] * len(x)
                for label, color in zip(comp_labels, comp_colors):
                    vals = comp_data[label]
                    if vals and len(vals) == len(x):
                        # Normalize each component's contribution by weight
                        ax2.fill_between(x, bottom,
                                         [b + v * 0.2 for b, v in zip(bottom, vals)],
                                         alpha=0.6, color=color, label=label)
                        bottom = [b + v * 0.2 for b, v in zip(bottom, vals)]
                ax2.set_title("Reward Components (last 20 steps)", color="#2e6a80", fontsize=8, pad=6)
                ax2.set_xlabel("Steps", color="#2e6a80", fontsize=7)
                ax2.tick_params(colors="#2e6a80", labelsize=6)
                ax2.grid(color="#0d2535", linewidth=0.5, linestyle="--")
                for spine in ax2.spines.values():
                    spine.set_edgecolor("#0d2535")
                ax2.legend(loc="upper right", fontsize=6,
                           facecolor="#080c12", edgecolor="#2e6a80", labelcolor="#c9d4e0")
                fig2.tight_layout(pad=1.0)
                st.pyplot(fig2)
                plt.close(fig2)

        # Decision Distribution
        if st.session_state.log:
            st.subheader("📊 Decision Distribution")
            decisions = [entry[1] for entry in st.session_state.log]
            from collections import Counter
            counts = dict(Counter(decisions))
            st.bar_chart(counts)

    # Episode Done banner
    if st.session_state.episode_done:
        final = st.session_state.score
        if final >= 0.6:
            st.success(f"🏆 EPISODE COMPLETE — Score: {final:.4f} — SUCCESS!")
        else:
            st.warning(f"⚠️ EPISODE COMPLETE — Score: {final:.4f} — Below success threshold (0.60)")


# ---------------------------
# EXECUTION
# ---------------------------
if "run_once" in st.session_state and st.session_state.run_once:
    result = run_step()
    if result:
        display(result)
    st.session_state.run_once = False

elif st.session_state.running:
    placeholder = st.empty()
    for _ in range(1000):
        if not st.session_state.running:
            break
        if st.session_state.episode_done:
            st.session_state.running = False
            break
        result = run_step()
        if result:
            with placeholder.container():
                display(result)
        time.sleep(speed)

elif st.session_state.last_result:
    display(st.session_state.last_result)

else:
    st.markdown("""
    <div class="idle">
        [ SYSTEM IDLE ]<br><br>
        SELECT A TASK IN THE SIDEBAR &nbsp; · &nbsp; PRESS &nbsp; ▶ RUN STEP &nbsp; OR &nbsp; ⚡ AUTO RUN &nbsp; TO BEGIN
    </div>
    """, unsafe_allow_html=True)