import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", max_episode_steps=300)

# Create and train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000000)
model.save("ppo_lunarlander")