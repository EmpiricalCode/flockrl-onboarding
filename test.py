import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_lunarlander")
env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(5):
    
    obs, info = env.reset()

    total_reward = 0.0

    while True:

        action, _ = model.predict(obs, deterministic=True)
        
        if hasattr(action, "item"):
            action = action.item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        env.render()  # should display in a window when render_mode="human"

        if terminated or truncated:
            print(f"Episode {episode + 1} total reward: {total_reward}")
            break
env.close()