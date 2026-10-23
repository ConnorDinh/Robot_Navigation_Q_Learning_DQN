from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent
from pathlib import Path
import pandas as pd
import torch

env = RobotNavigationEnv()
# agent = DQNAgent()
agent = DQNAgent(double_dqn=True)
successes = []
collisions = []
steps = []
rewards = []
global_steps = 0
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 100000
for episode in range(2000):
    state, info = env.reset()
    episode_steps = 0
    episode_collisions = 0
    episode_reward = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.buffer.push(state, action, reward, next_state, terminated)
        agent.update()
        episode_steps += 1
        global_steps += 1
        agent.epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * global_steps / epsilon_decay_steps)
        episode_reward += reward
        if global_steps % agent.update_frequency == 0:
            agent.target_network.load_state_dict(agent.policy_network.state_dict())
        if info["collision"]:
            episode_collisions += 1
        state = next_state
        if terminated or truncated:
            break
    rewards.append(episode_reward)
    successes.append(terminated)
    steps.append(episode_steps)
    collisions.append(episode_collisions)
    if (episode + 1) % 100 == 0:
        window = 100
        print(f"\nEpisode: {episode + 1}")
        print(f"Global steps: {global_steps}")
        print(f"Epsilon: {agent.epsilon:.4f}")
        print(f"Mean reward: {sum(rewards[-window:]) / window:.2f}")
        print(f"Success rate: {100 * sum(successes[-window:]) / window:.1f}%")
        print(f"Mean steps: {sum(steps[-window:]) / window:.2f}")
        print(f"Mean collisions: {sum(collisions[-window:]) / window:.2f}")
    if (episode + 1) % 500 == 0:
        Path("results/models").mkdir(parents=True, exist_ok=True)
        torch.save(agent.policy_network.state_dict(), f"results/models/double_dqn_history5_ep{episode + 1}.pt")

print("\nDouble DQN smoke test")
print(f"Episodes: {len(successes)}")
print(f"Success rate: {100 * sum(successes) / len(successes):.1f}%")
print(f"Mean reward: {sum(rewards) / len(rewards):.2f}")
print(f"Mean steps: {sum(steps) / len(steps):.2f}")

Path("results/models").mkdir(parents=True, exist_ok=True)
torch.save(agent.policy_network.state_dict(), "results/models/double_dqn_history5_weights.pt")

Path("results/logs").mkdir(parents=True, exist_ok=True)
df = pd.DataFrame({
    "episode": range(1, len(rewards) + 1),
    "reward": rewards,
    "steps": steps,
    "successes": successes,
    "collisions": collisions
})

df.to_csv("results/logs/double_dqn_history5_training.csv", index=False)