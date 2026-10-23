from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent

import numpy as np
import torch

env = RobotNavigationEnv()
agent = DQNAgent()

agent.policy_network.load_state_dict(torch.load("results/models/dqn_weights.pt",map_location="cpu",weights_only=True))

agent.policy_network.eval()
agent.epsilon = 0.0

rewards = []
steps = []
collisions = []
successes = []
successful_steps = []

for episode in range(100):
    state, info = env.reset(seed=42 + episode)
    visited_positions = {}
    start_position = tuple(env.robot_position)
    goal_position = tuple(env.goal_position)
    episode_reward = 0
    episode_steps = 0
    episode_collisions = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.select_action(state)
        position = tuple(env.robot_position)

        visited_positions[position] = (visited_positions.get(position, 0) + 1)
        next_state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_steps += 1
        episode_collisions += int(info["collision"])

        state = next_state

    rewards.append(episode_reward)
    steps.append(episode_steps)
    collisions.append(episode_collisions)
    successes.append(int(terminated))

    if terminated:
        successful_steps.append(episode_steps)
    else:
        repeated = sorted(visited_positions.items(), key=lambda item: item[1], reverse=True)

        print(f"\nFailed episode {episode + 1}")
        print(f"Start: {start_position}")
        print(f"Goal: {goal_position}")
        print(f"Collisions: {episode_collisions}")
        print(f"Most visited positions: {repeated[:5]}")

print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Success rate: {np.mean(successes) * 100:.1f}%")
print(f"Mean steps: {np.mean(steps):.2f}")
print(f"Mean collisions: {np.mean(collisions):.2f}")

if successful_steps:
    print(f"Mean successful steps: " f"{np.mean(successful_steps):.2f}")

env.close()