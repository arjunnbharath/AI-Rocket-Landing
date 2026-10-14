
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# Load the newer trained model
model = PPO.load("ppo_lunarlander_v2")

# Evaluate without displaying the game window
env = gym.make("LunarLander-v3")

rewards = []
successful_landings = 0

print("Evaluating the trained agent over 10 episodes...\n")

for episode in range(10):
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

    rewards.append(total_reward)

    # A positive reward is a useful rough signal, not a
    # definitive success detector.
    if total_reward >= 200:
        successful_landings += 1

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, "
        f"Steps = {steps}, "
        f"Terminated = {terminated}, "
        f"Truncated = {truncated}"
    )

env.close()

print("\n========== EVALUATION RESULTS ==========")
print(f"Average reward: {np.mean(rewards):.2f}")
print(f"Best reward:    {np.max(rewards):.2f}")
print(f"Worst reward:   {np.min(rewards):.2f}")
print(f"Episodes with reward >= 200: {successful_landings}/10")
