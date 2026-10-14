
import gymnasium as gym
from stable_baselines3 import PPO

# Load the existing trained model
model = PPO.load("ppo_lunarlander")

# Create the training environment
env = gym.make("LunarLander-v3")
model.set_env(env)

print("Continuing PPO training for 400,000 more steps...")

model.learn(
    total_timesteps=400_000,
    reset_num_timesteps=False,
)

model.save("ppo_lunarlander_v2")
env.close()

print("Training complete!")
print("Saved model: ppo_lunarlander_v2.zip")
