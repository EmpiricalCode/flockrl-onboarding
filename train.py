import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

# Create and train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)
model.save("ppo_lunarlander")