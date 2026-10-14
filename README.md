# 🚀 AI Rocket Landing using Reinforcement Learning

An AI agent trained to land a spacecraft safely using Reinforcement Learning (RL). This project uses Proximal Policy Optimization (PPO) to teach an agent how to control a lunar lander through trial and error in the Gymnasium LunarLander environment.

## 🎯 Project Overview

The goal is to train an AI agent that learns to land a lunar spacecraft safely by interacting with a simulated environment and maximizing its cumulative reward.

Instead of manually programming the landing strategy, the agent learns which actions to take based on rewards and penalties received during training.

## 🧠 Technologies Used

- Python 3.12
- Gymnasium — Reinforcement learning environment
- Stable-Baselines3 — Reinforcement learning library
- PPO (Proximal Policy Optimization) — Training algorithm
- PyTorch — Deep learning framework
- MLP (Multi-Layer Perceptron) — Neural network policy

## ⚙️ How It Works

1. The agent observes the spacecraft's current state.
2. It selects an action using its neural network policy.
3. The environment executes the action and returns a reward.
4. PPO updates the policy based on experience.
5. This process repeats as the agent learns a landing strategy.

### Observation Space

The environment provides 8 values describing the lander's state, including its position, velocity, angle, angular velocity, and leg-contact information.

### Action Space

The agent chooses from 4 discrete actions:

- Do nothing
- Fire the left orientation engine
- Fire the main engine
- Fire the right orientation engine

## 📊 Training Details

| Parameter | Value |
|---|---|
| Environment | `LunarLander-v3` |
| Algorithm | PPO |
| Policy | `MlpPolicy` |
| Initial training | 100,000 timesteps |
| Additional training | 400,000 timesteps |
| Total training | 500,000 timesteps |
| Evaluation episodes | 10 |

The model was trained from scratch without a pretrained model.

## 📈 Evaluation Results

The trained agent was evaluated across 10 episodes.

| Metric | Result |
|---|---:|
| Average reward | 207.68 |
| Best reward | 264.36 |
| Lowest reward | 0.57 |
| Episodes with reward ≥ 200 | 8/10 |

The results indicate that the agent has learned a promising landing strategy, although its performance is not yet consistent across every episode.

## 📁 Project Structure

```text
AI-Rocket-Landing/
├── main.py                 # Test the LunarLander environment
├── train_agent.py          # Train the PPO agent
├── continue_training.py    # Continue training the model
├── evaluate_agent.py       # Evaluate model performance
├── watch_agent.py          # Visually watch the trained agent
├── requirements.txt        # Project dependencies
├── ppo_lunarlander.zip     # Initial trained model
├── ppo_lunarlander_v2.zip  # Updated trained model
├── .gitignore              # Files excluded from Git
└── README.md               # Project documentation
```

## 🛠️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/arjunnbharath/AI-Rocket-Landing.git
cd AI-Rocket-Landing
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv venv312
```

### 3. Activate the virtual environment

Windows:

```bash
venv312\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the environment

```bash
python main.py
```

### 6. Train the agent

```bash
python train_agent.py
```

This trains the agent and saves the initial model.

### 7. Continue training

```bash
python continue_training.py
```

This loads the initial model, continues training, and saves the updated model.

### 8. Evaluate the trained agent

```bash
python evaluate_agent.py
```

This evaluates the saved model across multiple episodes and reports its performance.

### 9. Watch the agent in action

```bash
python watch_agent.py
```

This opens the LunarLander simulation so you can observe the trained agent.

## 🔮 Future Improvements

- Train the agent for more timesteps.
- Improve landing consistency across episodes.
- Compare PPO with other suitable reinforcement learning algorithms.
- Track training rewards using TensorBoard.
- Experiment with learning rates and batch sizes.
- Visualize training and evaluation results.
- Save the best-performing model based on evaluation rewards.

## 📚 Key Learnings

Through this project, I explored:

- Reinforcement learning fundamentals
- Agent–environment interaction
- Observations, actions, and rewards
- PPO and policy optimization
- Neural network policies
- Training, saving, loading, and evaluating AI models

## 👨‍💻 Author

**Arjun Bharath SR**

GitHub: [@arjunnbharath](https://github.com/arjunnbharath)

---

*Built as a hands-on project to explore Reinforcement Learning and AI agent development.*