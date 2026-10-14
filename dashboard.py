
from pathlib import Path

import gymnasium as gym
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import PPO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Rocket Landing | Mission Control",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# SPACE / MISSION CONTROL THEME
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #05080d;
        --panel: #0b111a;
        --panel2: #101925;
        --border: #253344;
        --text: #f2f6fc;
        --muted: #93a4b8;
        --blue: #4da3ff;
        --green: #47dfad;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 70% 0%, #10223a 0%, transparent 38%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(5, 8, 13, 0.8);
    }

    [data-testid="stSidebar"] {
        background: #080e16;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        letter-spacing: 0.5px;
        color: var(--text);
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 700 !important;
    }

    .eyebrow {
        color: var(--blue);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 2.4px;
        text-transform: uppercase;
    }

    .hero {
        background: linear-gradient(115deg, #111d2c, #090e16 72%);
        border: 1px solid #2a4057;
        border-radius: 14px;
        padding: 28px 30px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "MISSION CONTROL";
        position: absolute;
        right: 22px;
        top: 20px;
        color: #3a526d;
        letter-spacing: 3px;
        font-size: 0.65rem;
        font-weight: 700;
    }

    .hero-subtitle {
        color: #a8bad0;
        font-size: 0.92rem;
        line-height: 1.8;
    }

    .status {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        letter-spacing: 1px;
        font-weight: 700;
        background: #102c28;
        color: var(--green);
        border: 1px solid #24584b;
    }

    .metric-card {
        background: linear-gradient(145deg, #101a27, #0a1018);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px;
        min-height: 120px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.69rem;
        font-weight: 600;
        letter-spacing: 1.4px;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 10px;
        font-family: 'Barlow Condensed', sans-serif;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.74rem;
        margin-top: 5px;
    }

    .section-label {
        color: var(--blue);
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px;
    }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid #1c2b3c;
        border-radius: 8px;
        overflow: hidden;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: #1677e8;
        color: white;
        border: 1px solid #318cff;
        border-radius: 5px;
        font-weight: 600;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #318cff;
        color: white;
        border-color: #70b4ff;
    }

    [data-testid="stMetric"] {
        background: #0c141f;
        border: 1px solid var(--border);
        padding: 14px;
        border-radius: 8px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        color: #687b91;
        font-size: 0.72rem;
        padding-top: 20px;
        border-top: 1px solid var(--border);
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# HELPERS
# =====================================================
MODEL_FILES = {
    "PPO · Initial · 100K": "ppo_lunarlander.zip",
    "PPO · Extended · 500K": "ppo_lunarlander_v2.zip",
    "PPO · TensorBoard · 100K": "ppo_lunarlander_tb.zip",
}


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=320):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#a9bbd0"),
        margin=dict(l=12, r=12, t=48, b=12),
        title_font=dict(size=16, color="#f2f6fc"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            gridcolor="#1c2b3c",
            zerolinecolor="#33475c",
        ),
        yaxis=dict(
            gridcolor="#1c2b3c",
            zerolinecolor="#33475c",
        ),
    )
    return fig


def find_tensorboard_events():
    log_dir = Path("logs/tensorboard")
    return list(log_dir.rglob("events.out.tfevents.*")) if log_dir.exists() else []


def load_training_rewards():
    """Read mean episode rewards from TensorBoard, if available."""
    event_files = find_tensorboard_events()
    if not event_files:
        return None

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        points = []

        for event_file in event_files:
            accumulator = EventAccumulator(str(event_file))
            accumulator.Reload()

            tags = accumulator.Tags().get("scalars", [])

            reward_tags = [
                tag for tag in tags
                if "ep_rew_mean" in tag.lower()
            ]

            for tag in reward_tags:
                for item in accumulator.Scalars(tag):
                    points.append({
                        "Timesteps": item.step,
                        "Mean Episode Reward": item.value,
                        "Metric": tag,
                    })

        if not points:
            return None

        return pd.DataFrame(points).sort_values("Timesteps")

    except Exception:
        return None


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 22px 0">
            <div style="font-size:1.4rem;font-weight:700;
                        letter-spacing:4px">A R L <span style="color:#4da3ff">/</span></div>
            <div style="color:#93a4b8;font-size:0.69rem;
                        letter-spacing:2px;margin-top:5px">
                AUTONOMOUS LANDING
            </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="eyebrow">Mission Configuration</div>',
                unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Active model",
        list(MODEL_FILES.keys()),
        index=1,
    )

    episode_count = st.slider(
        "Evaluation episodes",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    st.markdown("---")
    st.markdown('<div class="eyebrow">System Information</div>',
                unsafe_allow_html=True)
    st.caption("ENVIRONMENT  /  LunarLander-v3")
    st.caption("ALGORITHM  /  PPO")
    st.caption("POLICY  /  MlpPolicy")
    st.caption("FRAMEWORK  /  Stable-Baselines3")

    st.markdown("---")
    st.caption("AI ROCKET LANDING")
    st.caption("Research Simulation · v1.0")

model_path = MODEL_FILES[selected_model]
model_exists = Path(model_path).exists()

# =====================================================
# HERO
# =====================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Autonomous Flight Systems · Research</div>
        <h1 style="margin:10px 0 4px 0">AI ROCKET LANDING</h1>
        <div class="hero-subtitle">
            Reinforcement learning mission control<br>
            PPO &nbsp; / &nbsp; LunarLander-v3 &nbsp; / &nbsp; PyTorch
        </div>
        <div style="margin-top:20px">
            <span class="status">● SIMULATION MODE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_exists:
    st.error(f"Model file not found: `{model_path}`")

# =====================================================
# TOP METRICS
# =====================================================
st.markdown('<div class="section-label">01 / FLIGHT SUMMARY</div>',
            unsafe_allow_html=True)

previous = st.session_state.get("evaluation_results")
previous_model = st.session_state.get("evaluation_model")

if previous is not None and previous_model == selected_model:
    avg_reward = previous["Reward"].mean()
    best_reward = previous["Reward"].max()
    avg_steps = previous["Steps"].mean()
    episode_count_done = len(previous)
else:
    avg_reward = None
    best_reward = None
    avg_steps = None
    episode_count_done = 0

events = find_tensorboard_events()

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "Average Reward",
        f"{avg_reward:.1f}" if avg_reward is not None else "—",
        "Latest evaluation",
    )
with c2:
    metric_card(
        "Best Reward",
        f"{best_reward:.1f}" if best_reward is not None else "—",
        "Best evaluated episode",
    )
with c3:
    metric_card(
        "Mean Episode Length",
        f"{avg_steps:.0f} steps" if avg_steps is not None else "—",
        "Across evaluated episodes",
    )
with c4:
    metric_card(
        "Training Logs",
        f"{len(events)} files",
        "TensorBoard event files found",
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# MAIN ACTIONS
# =====================================================
st.markdown('<div class="section-label">02 / MODEL OPERATIONS</div>',
            unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

with left:
    st.markdown("### Evaluation Control")
    st.write(
        f"Run a deterministic evaluation of **{selected_model}** "
        f"over **{episode_count} episodes**."
    )
    st.caption(
        "Evaluation runs in the LunarLander environment without rendering "
        "a game window."
    )

    run_evaluation = st.button(
        "▶  INITIATE EVALUATION",
        type="primary",
        use_container_width=True,
        disabled=not model_exists,
    )

with right:
    st.markdown("### Model Status")
    st.markdown(
        f"""
        <div class="panel">
            <div class="metric-label">SELECTED MODEL</div>
            <div style="font-size:1.05rem;font-weight:600;
                        margin:9px 0">{selected_model}</div>
            <div style="color:#93a4b8;font-size:0.8rem">
                File: {model_path}
            </div>
            <div style="margin-top:14px;color:{'#47dfad' if model_exists else '#ff7373'}">
                {'● MODEL FILE AVAILABLE' if model_exists else '● MODEL FILE MISSING'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_evaluation:
    rewards = []
    lengths = []
    statuses = []
    progress = st.progress(0)
    status_text = st.empty()

    try:
        model = PPO.load(model_path)
        env = gym.make("LunarLander-v3")

        try:
            for episode in range(episode_count):
                observation, info = env.reset()
                total_reward = 0.0
                steps = 0

                while True:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                    observation, reward, terminated, truncated, info = (
                        env.step(action)
                    )

                    total_reward += float(reward)
                    steps += 1

                    if terminated or truncated:
                        break

                rewards.append(round(total_reward, 2))
                lengths.append(steps)

                if truncated:
                    episode_status = "Time limit"
                elif terminated and total_reward >= 200:
                    episode_status = "High reward"
                elif terminated:
                    episode_status = "Terminated"
                else:
                    episode_status = "Unknown"

                statuses.append(episode_status)
                progress.progress((episode + 1) / episode_count)
                status_text.write(
                    f"Episode {episode + 1}/{episode_count} complete"
                )

        finally:
            env.close()

        result_df = pd.DataFrame({
            "Episode": range(1, episode_count + 1),
            "Reward": rewards,
            "Steps": lengths,
            "Status": statuses,
        })

        st.session_state["evaluation_results"] = result_df
        st.session_state["evaluation_model"] = selected_model
        st.success("Evaluation run completed.")

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")

# Refresh displayed results after a run
results = st.session_state.get("evaluation_results")
results_model = st.session_state.get("evaluation_model")

# =====================================================
# EVALUATION ANALYTICS
# =====================================================
st.markdown("---")
st.markdown('<div class="section-label">03 / TELEMETRY & ANALYTICS</div>',
            unsafe_allow_html=True)
st.markdown("### Evaluation Telemetry")

if results is not None:
    st.caption(f"Most recent run · {results_model}")

    fig_reward = px.line(
        results,
        x="Episode",
        y="Reward",
        markers=True,
        title="Reward by Episode",
    )
    fig_reward.add_hline(
        y=200,
        line_dash="dash",
        line_color="#47dfad",
        annotation_text="Reward reference: 200",
    )
    st.plotly_chart(
        style_chart(fig_reward, 340),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_steps = px.bar(
            results,
            x="Episode",
            y="Steps",
            title="Episode Duration",
            color="Steps",
            color_continuous_scale=["#183c61", "#4da3ff"],
        )
        st.plotly_chart(
            style_chart(fig_steps, 300),
            use_container_width=True,
        )

    with col_b:
        fig_dist = px.histogram(
            results,
            x="Reward",
            nbins=15,
            title="Reward Distribution",
            color_discrete_sequence=["#4da3ff"],
        )
        st.plotly_chart(
            style_chart(fig_dist, 300),
            use_container_width=True,
        )

    st.markdown("### Episode Records")
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button(
        "↓  EXPORT EVALUATION CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="rocket_landing_evaluation.csv",
        mime="text/csv",
    )

    st.caption(
        "A reward of 200 or more is only a reward-based indicator, "
        "not verified proof of a safe landing. Landing outcomes require "
        "separate state-based checks."
    )
else:
    st.info(
        "No evaluation results yet. Select a model and initiate an evaluation "
        "to populate the telemetry panels."
    )

# =====================================================
# TRAINING ANALYTICS
# =====================================================
st.markdown("---")
st.markdown('<div class="section-label">04 / TRAINING INTELLIGENCE</div>',
            unsafe_allow_html=True)
st.markdown("### PPO Training History")

training_rewards = load_training_rewards()

if training_rewards is not None:
    fig_training = px.line(
        training_rewards,
        x="Timesteps",
        y="Mean Episode Reward",
        color="Metric",
        title="Mean Episode Reward During Training",
    )
    st.plotly_chart(
        style_chart(fig_training, 360),
        use_container_width=True,
    )
    st.caption(
        "Values are read from available TensorBoard scalar logs. "
        "Different training runs may have separate metric series."
    )
elif events:
    st.warning(
        "TensorBoard event files exist, but no `ep_rew_mean` scalar was found. "
        "Other training metrics may still be available in TensorBoard."
    )
else:
    st.info(
        "No TensorBoard event files found. Run a training session with "
        "TensorBoard logging enabled to populate this section."
    )

st.markdown("#### TensorBoard Access")
st.code(
    "python -m tensorboard.main --logdir=logs/tensorboard",
    language="powershell",
)

st.link_button(
    "OPEN TENSORBOARD ↗",
    "http://localhost:6006/",
)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <div class="footer">
        AI ROCKET LANDING &nbsp; • &nbsp;
        PYTHON &nbsp; • &nbsp; GYMNASIUM &nbsp; • &nbsp;
        STABLE-BASELINES3 &nbsp; • &nbsp; STREAMLIT
        <br><br>
        Independent reinforcement learning project.
        Aerospace-inspired interface; not affiliated with SpaceX.
    </div>
    """,
    unsafe_allow_html=True,
)
