
import torch

from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent


MODEL_PATH = "results/models/double_dqn_history5_ep2000.pt"
NUM_EPISODES = 100


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

    successes = 0
    efficiencies = []
    extra_steps = []
    optimal_paths = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=42 + episode)

        start = tuple(env.robot_position)
        goal = tuple(env.goal_position)

        shortest_steps = env.shortest_path_length(start, goal)

        for step in range(env.max_steps):
            action = agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        actual_steps = step + 1

        if terminated:
            successes += 1

            efficiency = shortest_steps / actual_steps
            efficiencies.append(efficiency)

            extra_steps.append(actual_steps - shortest_steps)

            if actual_steps == shortest_steps:
                optimal_paths += 1

    env.close()

    print("\n===== PATH EFFICIENCY =====")
    print(f"Success rate: {100 * successes / NUM_EPISODES:.1f}%")

    if efficiencies:
        print(
            f"Mean path efficiency: "
            f"{100 * sum(efficiencies) / len(efficiencies):.2f}%"
        )
        print(
            f"Mean extra steps: "
            f"{sum(extra_steps) / len(extra_steps):.2f}"
        )
        print(f"Optimal paths: {optimal_paths}/{successes}")
        print(
            f"Optimal path rate (all episodes): "
            f"{100 * optimal_paths / NUM_EPISODES:.1f}%"
        )


if __name__ == "__main__":
    main()
