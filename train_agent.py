
import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the training environment
env = gym.make("LunarLander-v3")

# 2. Create the PPO agent with TensorBoard logging
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/tensorboard/",
)

# 3. Train the agent
print("Training started...")

model.learn(
    total_timesteps=100_000,
    tb_log_name="PPO_LunarLander",
)

# 4. Save the new trained model
model.save("ppo_lunarlander_tb")
env.close()

print("\nTraining complete!")
print("Model saved as ppo_lunarlander_tb.zip")
print("TensorBoard logs saved in logs/tensorboard/")

# 5. Test the trained agent visually
play_env = gym.make(
    "LunarLander-v3",
    render_mode="human",
)

for episode in range(3):
    observation, info = play_env.reset()
    total_reward = 0
    steps = 0

    while True:
        action, _states = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            play_env.step(action)
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

play_env.close()
print("Testing complete!")
