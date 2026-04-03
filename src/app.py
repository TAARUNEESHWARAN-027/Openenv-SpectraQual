import streamlit as st
import matplotlib.pyplot as plt
import time

from env import generate_pcb, update_factory, factory
from agent import get_state, choose_action
from reward import calculate_reward

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="SpectraQual",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# SESSION STATE
# ---------------------------
if "score" not in st.session_state:
    st.session_state.score = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False

# ---------------------------
# GLOBAL STYLES
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;600;700&family=Exo+2:wght@300;400;600;800&display=swap');

/* Base */
.stApp {
    background-color: #080c12;
    color: #c9d4e0;
    font-family: 'Exo 2', sans-serif;
}

/* Scanline overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.025) 0px, rgba(0,0,0,0.025) 1px,
        transparent 1px, transparent 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* Title */
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

/* Subheaders */
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

/* Metric cards */
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

/* Buttons */
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

/* Slider */
[data-testid="stSlider"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #2e6a80 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Alerts */
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

/* PCB card */
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

/* Slot grid */
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
.free { color:#00e676; }
.busy { color:#ff5a5a; }

/* Reward pill */
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

/* Score display */
.score-big {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-shadow: 0 0 14px currentColor;
}

/* Divider */
hr { border:none; border-top:1px solid rgba(0,229,255,0.08) !important; margin:1.2rem 0 !important; }

/* Idle banner */
.idle {
    text-align:center; padding:44px 20px;
    border:1px dashed rgba(0,229,255,0.15); border-radius:12px;
    color:#1e4a5a; font-family:'Share Tech Mono',monospace;
    font-size:0.8rem; letter-spacing:0.12em; margin-top:36px;
}

/* Pulse glow on decision alerts */
@keyframes pulseGlow {
    0%   { box-shadow: 0 0 5px rgba(0,229,255,0.15); }
    50%  { box-shadow: 0 0 22px rgba(0,229,255,0.45); }
    100% { box-shadow: 0 0 5px rgba(0,229,255,0.15); }
}
.stSuccess, .stWarning, .stError, .stInfo {
    animation: pulseGlow 1.5s ease-in-out infinite;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0d5e70, #00e5ff) !important;
    border-radius: 4px;
}
[data-testid="stProgressBar"] {
    background: #0a1420 !important;
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 4px;
}

/* Caption text */
.stCaption {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #2e6a80 !important;
    letter-spacing: 0.1em;
}

/* Bar chart */
[data-testid="stVegaLiteChart"] {
    background: #0a1420 !important;
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 8px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

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

def explain_decision(pcb, decision):
    if pcb["defect_type"] == "none":
        return "No defect detected — board cleared to PASS"
    if pcb["defect_type"] == "missing_component":
        return "High component cost → ROUTE to repair preferred" if decision.startswith("ROUTE") else "Low component cost → SCRAP is more economical"
    if pcb["defect_type"] == "solder_bridge":
        return "Repair recovers value if slot available, else system waits for capacity"
    if pcb["defect_type"] == "short_circuit":
        return "High risk defect → SCRAP is safer" if decision == "SCRAP" else "Low-risk profile → diagnostic route possible"
    return "Default decision rule applied"

# ---------------------------
# HEADER
# ---------------------------
st.title("⚔️ SPECTRAQUAL — SMART PCB DECISION SYSTEM")
st.markdown(
    '<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;'
    'color:#1e4a5a;letter-spacing:0.16em;margin-top:-10px;margin-bottom:4px;">'
    'REAL-TIME QUALITY INTELLIGENCE ENGINE · v2.1</p>',
    unsafe_allow_html=True
)

# ---------------------------
# SPEED SLIDER
# ---------------------------
speed = st.slider("⚡ Simulation Speed (seconds per step)", 0.2, 2.0, 1.0, step=0.1)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# METRICS
# ---------------------------
m1, m2, m3 = st.columns(3)
m1.metric("💰 Total Score",     round(st.session_state.score, 2))
m2.metric("⚙️ Active Slots",    sum(1 for s in factory["soldering_slots"] if s > 0))
m3.metric("🧠 Decisions Taken", len(st.session_state.history))

# LIVE STATUS BAR
last_score = round(st.session_state.history[-1], 2) if st.session_state.history else "N/A"
status_color = "#00e5ff" if st.session_state.history else "#1e4a5a"
st.markdown(f"""
<div style="
    font-family:'Share Tech Mono',monospace;
    font-size:0.75rem;
    color:{status_color};
    padding:6px 14px;
    border:1px solid rgba(0,229,255,0.2);
    border-radius:6px;
    display:inline-block;
    margin-top:10px;
    margin-bottom:4px;
    background:rgba(0,229,255,0.03);
    letter-spacing:0.1em;
">
🟢 SYSTEM {'ACTIVE' if st.session_state.history else 'STANDBY'} &nbsp;·&nbsp; LAST SCORE: {last_score} &nbsp;·&nbsp; STEPS: {len(st.session_state.history)}
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# CONTROL BUTTONS
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("▶  RUN STEP"):
        st.session_state.running = False
        st.session_state.run_once = True
with c2:
    if st.button("⚡  AUTO RUN"):
        st.session_state.running = True
with c3:
    if st.button("⛔  STOP"):
        st.session_state.running = False
with c4:
    if st.button("🔄  RESET"):
        st.session_state.score = 0
        st.session_state.history = []
        st.session_state.log = []
        factory["soldering_slots"] = [0, 0, 0]

# ---------------------------
# CORE STEP
# ---------------------------
def run_step():
    update_factory()
    pcb      = generate_pcb()
    state    = get_state(pcb, factory)
    decision = choose_action(state, epsilon=0)
    reward   = calculate_reward(pcb, decision)

    # AI Confidence — derived from board criticality
    confidence = round(min(0.6 + 0.4 * float(pcb.get("criticality", 0.5)), 1.0), 2)

    st.session_state.score += reward
    st.session_state.history.append(st.session_state.score)

    # Decision log for distribution chart
    if "log" not in st.session_state:
        st.session_state.log = []
    st.session_state.log.append((pcb, decision))

    return pcb, decision, reward, confidence

# ---------------------------
# DISPLAY
# ---------------------------
def display(pcb, decision, reward, confidence):
    from collections import Counter
    col1, col2 = st.columns(2, gap="large")

    # ── LEFT ──
    with col1:
        st.subheader("PCB Info")
        st.markdown(f"""
        <div class="pcb-card">
            <div><span class="lbl">Board ID &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                 <span class="val">{pcb.get('board_id', 'N/A')}</span></div>
            <div><span class="lbl">Defect Type &nbsp;&nbsp;</span>
                 {defect_badge(pcb['defect_type'])}</div>
            <div><span class="lbl">Component Cost </span>
                 <span class="val">₹{pcb.get('component_cost', '—')}</span></div>
            <div><span class="lbl">Criticality &nbsp;&nbsp;&nbsp;</span>
                 <span class="val">{pcb.get('criticality', '—')}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Decision")
        if decision == "PASS":
            st.success(f"✅  {decision}")
        elif "ROUTE" in decision:
            st.warning(f"🛠️  {decision}")
        elif decision == "WAIT":
            st.warning("⏳  WAITING FOR SLOT AVAILABILITY")
        else:
            st.error(f"❌  {decision}")

        st.subheader("AI Confidence")
        st.progress(confidence)
        st.caption(f"{int(confidence * 100)}% confidence")

        st.subheader("Reward")
        if reward > 0:
            st.markdown(f'<span class="rpill rpos">▲ +{round(reward,2)}</span>', unsafe_allow_html=True)
        elif reward < 0:
            st.markdown(f'<span class="rpill rneg">▼ {round(reward,2)}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="rpill rzero">● {reward}</span>', unsafe_allow_html=True)

        st.subheader("🧠 Why this decision?")
        st.info(explain_decision(pcb, decision))

    # ── RIGHT ──
    with col2:
        st.subheader("Factory Slots")
        slot_html = '<div class="slot-grid">'
        for i, slot in enumerate(factory["soldering_slots"]):
            if slot > 0:
                slot_html += (f'<div class="slot-item"><div class="dot dot-busy"></div>'
                              f'<span class="busy">SLOT {i:02d} · {slot}t</span></div>')
            else:
                slot_html += (f'<div class="slot-item"><div class="dot dot-free"></div>'
                              f'<span class="free">SLOT {i:02d} · FREE</span></div>')
        slot_html += '</div>'
        st.markdown(slot_html, unsafe_allow_html=True)

        st.subheader("Cumulative Score")
        score_color = "#00e676" if st.session_state.score >= 0 else "#ff5a5a"
        st.markdown(
            f'<div class="score-big" style="color:{score_color}">'
            f'{round(st.session_state.score, 2)}</div>',
            unsafe_allow_html=True
        )

        st.subheader("📈 Profit Trend")
        fig, ax = plt.subplots(figsize=(5.5, 3))
        fig.patch.set_facecolor("#080c12")
        ax.set_facecolor("#0a1420")
        history = st.session_state.history
        if history:
            ax.plot(history, color="#00e5ff", linewidth=1.8,
                    marker='o', markersize=3.5,
                    markerfacecolor="#00e5ff", markeredgewidth=0)
            ax.fill_between(range(len(history)), history, alpha=0.10, color="#00e5ff")
        ax.set_title("System Performance", color="#2e6a80", fontsize=9, pad=8)
        ax.set_xlabel("Steps",  color="#2e6a80", fontsize=8)
        ax.set_ylabel("Score",  color="#2e6a80", fontsize=8)
        ax.tick_params(colors="#2e6a80", labelsize=7)
        ax.grid(color="#0d2535", linewidth=0.7, linestyle="--")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0d2535")
        fig.tight_layout(pad=1.2)
        st.pyplot(fig)
        plt.close(fig)

        # Decision Distribution
        if "log" in st.session_state and st.session_state.log:
            st.subheader("📊 Decision Distribution")
            decisions = [entry[1] for entry in st.session_state.log]
            counts = dict(Counter(decisions))
            st.bar_chart(counts)

# ---------------------------
# EXECUTION
# ---------------------------
if "run_once" in st.session_state and st.session_state.run_once:
    pcb, decision, reward, confidence = run_step()
    display(pcb, decision, reward, confidence)
    st.session_state.run_once = False

elif st.session_state.running:
    placeholder = st.empty()
    for _ in range(1000):
        if not st.session_state.running:
            break
        pcb, decision, reward, confidence = run_step()
        with placeholder.container():
            display(pcb, decision, reward, confidence)
        time.sleep(speed)

else:
    st.markdown("""
    <div class="idle">
        [ SYSTEM IDLE ]<br><br>
        PRESS &nbsp; ▶ RUN STEP &nbsp; OR &nbsp; ⚡ AUTO RUN &nbsp; TO BEGIN
    </div>
    """, unsafe_allow_html=True)