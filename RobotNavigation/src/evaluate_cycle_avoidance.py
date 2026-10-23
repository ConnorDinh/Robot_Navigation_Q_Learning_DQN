import torch
from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent

MODEL_PATH = "results/models/double_dqn_8feat_ep1500.pt"
NUM_EPISODES = 100

def choose_action(agent, env, state, previous_position):
    state_tensor = agent.preprocess_state(state)

    with torch.no_grad():
        q_values = agent.policy_network(state_tensor)

    # Consider actions in descending Q-value order
    ranked_actions = torch.argsort(q_values, descending=True).tolist()

    for action in ranked_actions:
        dr, dc = env.actions[action]

        next_position = (
            env.robot_position[0] + dr,
            env.robot_position[1] + dc
        )

        # Avoid invalid moves
        if not env.is_valid_position(next_position):
            continue

        # Avoid immediately returning to the previous position
        if next_position == previous_position:
            continue

        return action

    # Fallback: use the highest-valued valid action
    for action in ranked_actions:
        dr, dc = env.actions[action]

        next_position = (
            env.robot_position[0] + dr,
            env.robot_position[1] + dc
        )

        if env.is_valid_position(next_position):
            return action

    return ranked_actions[0]

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

    successes = []
    steps = []
    collisions = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=42 + episode)

        previous_position = None
        episode_collisions = 0

        for step in range(env.max_steps):
            current_position = tuple(env.robot_position)

            action = choose_action(
                agent,
                env,
                state,
                previous_position
            )

            state, reward, terminated, truncated, info = env.step(action)

            episode_collisions += int(info["collision"])
            previous_position = current_position

            if terminated or truncated:
                break

        successes.append(int(terminated))
        steps.append(step + 1)
        collisions.append(episode_collisions)

    env.close()

    print("Double DQN + cycle avoidance")
    print(f"Success rate: {100 * sum(successes) / NUM_EPISODES:.1f}%")
    print(f"Mean steps: {sum(steps) / NUM_EPISODES:.2f}")
    print(f"Mean collisions: {sum(collisions) / NUM_EPISODES:.2f}")


if __name__ == "__main__":
    main()
