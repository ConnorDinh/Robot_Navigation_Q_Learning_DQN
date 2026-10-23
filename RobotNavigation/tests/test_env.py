from src.environment import RobotNavigationEnv

env = RobotNavigationEnv(render_mode="human")

observation, info = env.reset(seed=42)

assert env.observation_space.contains(observation)

for _ in range(100):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(observation)
    assert env.robot_position not in env.obstacle_positions

    env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()

print("Environment test passed!")