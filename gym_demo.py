import gymnasium as gym
# Make environment (can get rid of render_mode arg to get rid of visualization)
env = gym.make("FrozenLake-v1", render_mode="human")
# Reset environment
observation, info = env.reset()
# Take random actions
for _ in range(100):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()