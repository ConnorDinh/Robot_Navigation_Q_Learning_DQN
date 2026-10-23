
from collections import Counter

import torch

from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent


MODEL_PATH = "results/models/double_dqn_history5_ep2000.pt"
NUM_EPISODES = 100


def detect_two_cell_cycle(path):
    """Check whether the final 20 moves alternate between two cells."""
    if len(path) < 20:
        return False

    recent = path[-20:]

    return all(
        recent[i] == recent[i - 2]
        for i in range(2, len(recent))
    ) and recent[0] != recent[1]


def detect_repeating_cycle(path, max_cycle_length=10):
    """Find a repeating cycle at the end of a trajectory."""
    for cycle_length in range(2, max_cycle_length + 1):
        if len(path) < 3 * cycle_length:
            continue

        a = path[-cycle_length:]
        b = path[-2 * cycle_length:-cycle_length]
        c = path[-3 * cycle_length:-2 * cycle_length]

        if a == b == c:
            return cycle_length

    return None


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

    failed_episodes = 0
    two_cell_cycles = 0
    longer_cycles = 0
    other_failures = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=42 + episode)

        start = tuple(env.robot_position)
        goal = tuple(env.goal_position)

        path = [start]

        for step in range(env.max_steps):
            action = agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)
            path.append(tuple(env.robot_position))

            if terminated or truncated:
                break

        if terminated:
            continue

        failed_episodes += 1

        cycle_length = detect_repeating_cycle(path)

        if detect_two_cell_cycle(path):
            failure_type = "Two-cell cycle"
            two_cell_cycles += 1
        elif cycle_length is not None:
            failure_type = f"Cycle of length {cycle_length}"
            longer_cycles += 1
        else:
            failure_type = "Other failure"
            other_failures += 1

        counts = Counter(path)

        print(f"\nFailed episode: {episode}")
        print(f"Start: {start}, Goal: {goal}")
        print(f"Failure type: {failure_type}")
        print(f"Unique positions: {len(counts)}")
        print(f"Most visited: {counts.most_common(3)}")
        print(f"Last 20 positions: {path[-20:]}")

    env.close()

    print("\n========== SUMMARY ==========")
    print(f"Total episodes: {NUM_EPISODES}")
    print(f"Successful: {NUM_EPISODES - failed_episodes}")
    print(f"Failed: {failed_episodes}")
    print(f"Two-cell cycles: {two_cell_cycles}")
    print(f"Longer cycles: {longer_cycles}")
    print(f"Other failures: {other_failures}")


if __name__ == "__main__":
    main()
