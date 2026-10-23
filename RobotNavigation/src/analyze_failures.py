from collections import Counter
import torch
from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent

MODEL_PATH = "results/models/double_dqn_8feat_ep1500.pt"

def main():
    env = RobotNavigationEnv()
    agent = DQNAgent(double_dqn=True)

    agent.policy_network.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    agent.policy_network.eval()
    agent.epsilon = 0.0

    failures_shown = 0

    for episode in range(100):
        state, _ = env.reset(seed=42 + episode)

        start = tuple(env.robot_position)
        goal = tuple(env.goal_position)

        path = [start]
        actions = []

        for step in range(env.max_steps):
            action = agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)

            path.append(tuple(env.robot_position))
            actions.append(action)

            if terminated or truncated:
                break

        if not terminated:
            counts = Counter(path)
            most_common = counts.most_common(5)

            print(f"\nFailed episode: {episode}")
            print(f"Start: {start}")
            print(f"Goal: {goal}")
            print(f"Steps: {len(actions)}")
            print(f"Unique positions: {len(counts)}")
            print(f"Most visited positions: {most_common}")
            print(f"First 30 positions: {path[:30]}")
            print(f"Last 20 positions: {path[-20:]}")
            print(f"Last 20 actions: {actions[-20:]}")

            failures_shown += 1

            if failures_shown >= 5:
                break

    env.close()


if __name__ == "__main__":
    main()
