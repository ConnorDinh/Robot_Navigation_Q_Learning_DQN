from pathlib import Path
import torch
from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent

CHECKPOINTS = [500, 1000, 1500, 2000]
NUM_EPISODES = 100

def evaluate_checkpoint(path):
    env = RobotNavigationEnv()
    agent = DQNAgent(double_dqn=True)

    agent.policy_network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    agent.policy_network.eval()
    agent.epsilon = 0.0

    successes = []
    rewards = []
    steps = []
    collisions = []
    successful_steps = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=42 + episode)
        episode_reward = 0
        episode_collisions = 0
        for step in range(env.max_steps):
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_collisions += int(info["collision"])
            if terminated or truncated:
                break
        successes.append(int(terminated))
        rewards.append(episode_reward)
        steps.append(step + 1)
        collisions.append(episode_collisions)
        if terminated:
            successful_steps.append(step + 1)
    env.close()

    return {
        "success": 100 * sum(successes) / NUM_EPISODES,
        "reward": sum(rewards) / NUM_EPISODES,
        "steps": sum(steps) / NUM_EPISODES,
        "collisions": sum(collisions) / NUM_EPISODES,
        "successful_steps": (
            sum(successful_steps) / len(successful_steps)
            if successful_steps else None
        ),
    }


def main():
    for episode in CHECKPOINTS:
        path = Path(f"results/models/double_dqn_simple_reward_ep{episode}.pt")

        if not path.exists():
            print(f"Missing checkpoint: {path}")
            continue

        results = evaluate_checkpoint(path)

        print(f"\nDouble DQN checkpoint: {episode}")
        print(f"Success rate: {results['success']:.1f}%")
        print(f"Mean reward: {results['reward']:.2f}")
        print(f"Mean steps: {results['steps']:.2f}")
        print(f"Mean collisions: {results['collisions']:.2f}")

        if results["successful_steps"] is not None:
            print(
                f"Mean successful steps: "
                f"{results['successful_steps']:.2f}"
            )


if __name__ == "__main__":
    main()
