from src.dqn_agent import DQNAgent
import numpy as np

def main():
    agent = DQNAgent()

    # Test 1: Action selection
    state = np.array([0, 0, 14, 14])
    action = agent.select_action(state)

    assert 0 <= action < 4
    print("Test 1 passed: Action selection")

    # Test 2: Replay buffer
    for _ in range(100):
        agent.buffer.push(
            [0, 0, 14, 14],
            0,
            -1.0,
            [0, 1, 14, 14],
            False
        )

    assert agent.buffer.length() == 100

    states, actions, rewards, next_states, dones = (
        agent.buffer.sample(64)
    )

    assert len(states) == 64
    assert len(actions) == 64
    assert len(rewards) == 64
    assert len(next_states) == 64
    assert len(dones) == 64

    print("Test 2 passed: Replay buffer")

    # Test 3: Neural network output
    import torch

    state_tensor = torch.tensor(
        state, dtype=torch.float32
    ) / 14.0

    q_values = agent.policy_network(state_tensor)

    assert q_values.shape == (4,)
    print("Test 3 passed: Q-Network")

    # Test 4: DQN update
    loss = agent.update()

    assert loss is not None
    assert np.isfinite(loss)

    print("Test 4 passed: DQN update")
    print("Loss:", loss)

    print("\nAll DQN tests passed!")


if __name__ == "__main__":
    main()