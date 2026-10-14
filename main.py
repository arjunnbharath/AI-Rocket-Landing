
import gymnasium as gym

# Create the LunarLander environment
env = gym.make("LunarLander-v3", render_mode="human")

# Start one episode
observation, info = env.reset()

print("Initial observation:", observation)
print("Observation shape:", observation.shape)
print("Available actions:", env.action_space)

for step in range(1000):
    # Choose a random action for now
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished!")
        print("Total steps:", step + 1)
        break

env.close()
