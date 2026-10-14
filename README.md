# 🚀 AI Rocket Landing

An autonomous lunar landing simulation powered by Reinforcement Learning. This project uses Proximal Policy Optimization (PPO) to train an AI agent to control a lunar lander in Gymnasium's `LunarLander-v3` environment.

The project includes a Streamlit mission-control dashboard for model evaluation, training analytics, reward visualization, and experiment monitoring.

## ✨ Features

- **PPO Agent Training:** Train an AI agent using Stable-Baselines3.
- **Mission-Control Dashboard:** Aerospace-inspired dark UI built with Streamlit.
- **Model Selection:** Load and evaluate different saved PPO models.
- **Evaluation Analytics:** Analyze episode rewards, episode lengths, and reward distributions.
- **Training Visualization:** Explore TensorBoard training logs and reward curves when available.
- **CSV Export:** Download evaluation results for further analysis.
- **Interactive Charts:** Visualize performance using Plotly.
- **Model Persistence:** Save and reload trained agents.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| Gymnasium | Lunar lander simulation |
| Stable-Baselines3 | Reinforcement learning algorithms |
| PPO | Policy optimization algorithm |
| PyTorch | Deep learning framework |
| Streamlit | Interactive dashboard |
| Plotly | Interactive data visualization |
| TensorBoard | Training metrics and experiment monitoring |
| Pandas | Evaluation data analysis |

## 🖥️ Dashboard

The Streamlit dashboard provides a centralized view of your AI agent's performance.

### Dashboard capabilities

- Select a saved PPO model.
- Run evaluations across multiple episodes.
- View average and best reward for the latest evaluation.
- Visualize episode rewards and episode lengths.
- Explore the reward distribution.
- Export evaluation results as CSV.
- Inspect TensorBoard training metrics when logged data is available.

The interface uses a dark, aerospace-inspired design with mission-control styling. It is an independent project and is not affiliated with SpaceX.

## ⚙️ Reinforcement Learning Setup

**Environment:** `LunarLander-v3`

**Algorithm:** Proximal Policy Optimization (PPO)

**Policy:** `MlpPolicy`

**Observation space:** 8 values describing the lander's state.

**Action space:** 4 discrete control actions.

The agent learns by interacting with the simulation, receiving rewards or penalties, and updating its policy during training.

## 📊 Previously Recorded Evaluation

The extended model was previously evaluated over 10 episodes.

| Metric | Recorded result |
|---|---:|
| Average reward | 207.68 |
| Best reward | 264.36 |
| Lowest reward | 0.57 |
| Episodes with reward ≥ 200 | 8/10 |
| Training timesteps | 500,000 |

These figures describe the earlier evaluation of `ppo_lunarlander_v2.zip`. They are not live dashboard metrics.

A reward of 200 or more is only a rough performance indicator and does not independently confirm a successful landing.

## 📁 Project Structure

```text
AI-Rocket-Landing/
├── dashboard.py              # Streamlit mission-control dashboard
├── main.py                   # Test the simulation environment
├── train_agent.py            # Train PPO with TensorBoard logging
├── continue_training.py      # Continue training a saved model
├── evaluate_agent.py         # Evaluate the trained agent
├── watch_agent.py            # Watch the agent in the simulation
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Excluded files
├── ppo_lunarlander.zip       # Initial model
├── ppo_lunarlander_v2.zip    # Extended-training model
├── ppo_lunarlander_tb.zip    # TensorBoard-enabled training model
└── logs/
    └── tensorboard/           # Training event logs, when generated
```

The trained model files and TensorBoard logs may be kept locally rather than committed to GitHub.

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/arjunnbharath/AI-Rocket-Landing.git
cd AI-Rocket-Landing
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.12 -m venv venv312
venv312\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Run the Dashboard

From the project root, run:

```bash
streamlit run dashboard.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

Select a model in the sidebar and click **INITIATE EVALUATION** to generate charts and metrics.

## 🏋️ Train the Agent

Start a fresh training run with TensorBoard logging:

```bash
python train_agent.py
```

The script trains the PPO agent for 100,000 timesteps, saves the model as `ppo_lunarlander_tb.zip`, and runs three visual test episodes.

**Note:** This script starts a new model rather than continuing from the existing 500,000-timestep model.

## 🔄 Continue Training an Existing Model

```bash
python continue_training.py
```

This loads the existing model configured in the script, continues training, and saves an updated checkpoint.

## 📈 View TensorBoard Metrics

Run TensorBoard in a separate terminal:

```bash
python -m tensorboard.main --logdir=logs/tensorboard
```

Open:

`http://localhost:6006`

TensorBoard displays the metrics actually recorded during logged training runs. Previous training history cannot be reconstructed if it was not logged.

## 🧪 Evaluate the Agent

Run the evaluation script:

```bash
python evaluate_agent.py
```

This evaluates the saved model and reports its performance across multiple episodes.

## 🎮 Watch the Agent

```bash
python watch_agent.py
```

This opens the LunarLander simulation so you can observe the trained agent's behavior.

## 🔮 Future Improvements

- Improve landing consistency across evaluation episodes.
- Add state-based landing success and crash detection.
- Compare PPO with other reinforcement learning algorithms.
- Visualize lander altitude, velocity, and angle over time.
- Add training-run comparisons and experiment history.
- Track model checkpoints and hyperparameters.
- Deploy the dashboard for public access.

## 📚 Key Learnings

This project explores:

- Reinforcement learning fundamentals.
- Agent–environment interactions.
- PPO and neural network policies.
- Training and evaluating AI agents.
- Saving and loading trained models.
- Experiment tracking with TensorBoard.
- Interactive dashboards and performance visualization.

## 👨‍💻 Author

**Arjun Bharath SR**

GitHub: [@arjunnbharath](https://github.com/arjunnbharath)

**Repository:** [AI-Rocket-Landing](https://github.com/arjunnbharath/AI-Rocket-Landing)

---

*An AI engineering project exploring reinforcement learning, autonomous control, and simulation-based experimentation.*