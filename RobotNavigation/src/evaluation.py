from src.environment import RobotNavigationEnv
from src.q_learning_agent import QLearningAgent
import numpy as np
import random

env = RobotNavigationEnv()
agent = QLearningAgent(15, 0.1, 0.99, 0.0)
agent.q_table = np.load("results/models/q_table.npy")
collisions = []
steps = []
rewards = []
successes = []

for episode in range(100):
    state, info = env.reset(seed=42+episode)
    start_position = env.robot_position
    episode_step = 0
    episode_reward = 0
    episode_collision = 0
    visited_positions = {}
    while True:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
        position = tuple(env.robot_position)
        visited_positions[position] = visited_positions.get(position, 0) + 1
        episode_step += 1
        episode_reward += reward
        if info["collision"]:
            episode_collision += 1
        state = observation
        if terminated or truncated:
            break
    successes.append(terminated)
    collisions.append(episode_collision)
    rewards.append(episode_reward)
    steps.append(episode_step)
    if not terminated:
        repeated = sorted(visited_positions.items(), key=lambda item: item[1],reverse=True)
        print(f"\nFailed episode {episode + 1}")
        print("Most visited positions:", repeated[:5])
mean_reward = sum(rewards) / len(rewards)
success_rate = sum(successes) / len(successes) * 100
mean_steps = sum(steps)/ len(steps)
mean_collisions = sum(collisions) / len(collisions)
print("Mean reward: ", mean_reward)
print("Success rate: ", success_rate, "%")
print("Mean steps: ", mean_steps)
print("Mean collisions: ", mean_collisions)

successful_steps = [steps[i] for i in range(len(steps)) if successes[i] == 1]

print("Mean successful steps:", np.mean(successful_steps))
print("Failure rate:", 100 - np.mean(successes) * 100, "%")
