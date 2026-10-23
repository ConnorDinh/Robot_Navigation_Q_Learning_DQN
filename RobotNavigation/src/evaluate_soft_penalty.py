
from collections import deque

import torch

from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent


MODEL_PATH = "results/models/double_dqn_history5_ep2000.pt"

NUM_EPISODES = 500
LAMBDAS = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
HISTORY_LENGTH = 5


def select_action_with_penalty(agent, env, state, recent_positions, lam):
    state_tensor = agent.preprocess_state(state)

    with torch.no_grad():
        q_values = agent.policy_network(state_tensor).flatten()

    adjusted_q = q_values.clone()

    for action in range(env.action_space.n):
        dr, dc = env.actions[action]

        next_position = (
            env.robot_position[0] + dr,
            env.robot_position[1] + dc
        )

        # Exclude invalid actions from consideration.
        if not env.is_valid_position(next_position):
            adjusted_q[action] = float("-inf")
            continue

        visits = recent_positions.count(next_position)
        adjusted_q[action] -= lam * visits

    return int(torch.argmax(adjusted_q).item())


def evaluate(agent, lam):
    env = RobotNavigationEnv()

    successes = 0
    total_steps = 0
    total_collisions = 0

    efficiencies = []
    optimal_paths = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=10000 + episode)

        start = tuple(env.robot_position)
        goal = tuple(env.goal_position)

        shortest_steps = env.shortest_path_length(start, goal)

        recent_positions = deque(
            [start],
            maxlen=HISTORY_LENGTH
        )

        for step in range(env.max_steps):
            current_position = tuple(env.robot_position)

            action = select_action_with_penalty(
                agent,
                env,
                state,
                recent_positions,
                lam
            )

            state, reward, terminated, truncated, info = env.step(action)

            recent_positions.append(tuple(env.robot_position))

            total_collisions += int(info["collision"])

            if terminated or truncated:
                break

        actual_steps = step + 1
        total_steps += actual_steps

        if terminated:
            successes += 1

            efficiencies.append(
                shortest_steps / actual_steps
            )

            if actual_steps == shortest_steps:
                optimal_paths += 1

    env.close()

    return {
        "success": 100 * successes / NUM_EPISODES,
        "steps": total_steps / NUM_EPISODES,
        "collisions": total_collisions / NUM_EPISODES,
        "efficiency": (
            100 * sum(efficiencies) / len(efficiencies)
            if efficiencies else 0
        ),
        "optimal_paths": optimal_paths
    }


def main():
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

    print("\n===== SOFT PENALTY EXPERIMENT =====")

    for lam in LAMBDAS:
        result = evaluate(agent, lam)

        print(f"\nLambda: {lam}")
        print(f"Success rate: {result['success']:.1f}%")
        print(f"Mean steps: {result['steps']:.2f}")
        print(f"Mean collisions: {result['collisions']:.2f}")
        print(f"Mean path efficiency: {result['efficiency']:.2f}%")
        print(f"Optimal paths: {result['optimal_paths']}/{NUM_EPISODES}")


if __name__ == "__main__":
    main()
