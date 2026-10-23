
import unittest

import torch

from src.environment import RobotNavigationEnv
from src.dqn_agent import DQNAgent


class TestDoubleDQNFeatures(unittest.TestCase):

    def test_observation_shape(self):
        env = RobotNavigationEnv()
        state, _ = env.reset(seed=42)

        self.assertEqual(len(state), 8)
        self.assertTrue(env.observation_space.contains(state))
        env.close()

    def test_reproducible_reset(self):
        env = RobotNavigationEnv()

        state1, _ = env.reset(seed=42)
        state2, _ = env.reset(seed=42)

        self.assertEqual(state1, state2)
        env.close()

    def test_network_output(self):
        agent = DQNAgent(double_dqn=True)

        state = [5, 6, 12, 10, 0, 1, 0, 0]
        processed = agent.preprocess_state(state)

        output = agent.policy_network(processed)

        self.assertEqual(tuple(output.shape), (4,))

    def test_double_dqn_update(self):
        env = RobotNavigationEnv()
        agent = DQNAgent(double_dqn=True)

        state, _ = env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.buffer.push(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            state = next_state

            if terminated or truncated:
                state, _ = env.reset()

        loss = agent.update()

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        env.close()


if __name__ == "__main__":
    unittest.main()
