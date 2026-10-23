from src.environment import RobotNavigationEnv
from src.q_learning_agent import QLearningAgent
import numpy as np
from pathlib import Path
import pandas as pd

env = RobotNavigationEnv()
agent = QLearningAgent(15, 0.1, 0.99, 1.0)
NUMBER_OF_EPISODES = 10000
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.05
rewards = []
steps = []
success = []
epsilons = []
for episode in range(NUMBER_OF_EPISODES):
    state, info = env.reset()

    episode_reward = 0
    episode_steps = 0

    while True:
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state, terminated)

        state = next_state
        episode_reward += reward
        episode_steps += 1
        if terminated or truncated:
            break
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        avg_steps = np.mean(steps[-100:])
        success_rate = np.mean(success[-100:]) * 100

        print(
            f"Episode {episode + 1}/{NUMBER_OF_EPISODES} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Steps: {avg_steps:.2f} | "
            f"Success Rate: {success_rate:.1f}% | "
            f"Epsilon: {agent.epsilon:.3f}"
        )
    rewards.append(episode_reward)
    steps.append(episode_steps)
    success.append(int(terminated))
    epsilons.append(agent.epsilon)
    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

Path("results/models").mkdir(parents=True, exist_ok=True)
np.save("results/models/q_table.npy", agent.q_table)

Path("results/logs").mkdir(parents=True, exist_ok=True)
df = pd.DataFrame({
    "episode": range(1, NUMBER_OF_EPISODES + 1),
    "reward": rewards,
    "steps": steps,
    "success": success,
    "epsilon": epsilons
})

df.to_csv("results/logs/q_learning_training.csv", index=False)