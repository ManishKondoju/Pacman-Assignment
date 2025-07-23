import gym

env = gym.make("ALE/Pacman-v5", render_mode="human")
obs, info = env.reset()  # Gym 0.26 returns obs, info

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())  # Returns 5 items
    if terminated or truncated:
        obs, info = env.reset()
env.close()
