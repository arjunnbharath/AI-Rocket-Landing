
import gymnasium as gym
from stable_baselines3 import PPO

# Load the trained agent
model = PPO.load("ppo_lunarlander_v2")

# Open the visible LunarLander environment
env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(3):
    observation, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, "
        f"Steps = {steps}"
    )

env.close()
