import gymnasium as gym
from collections import deque
import pygame

class RobotNavigationEnv(gym.Env):
    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=15, max_steps=200):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.history_length = 5
        self.recent_positions = deque(maxlen=self.history_length)
        self.actions = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)} 
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete([self.grid_size] * 4 + [2] * 4 + [self.history_length+1] * 4)

        self.obstacle_positions = set()
        self.initialize_obstacle()
        self.robot_position = None
        self.goal_position = None
        self.step_count = 0

        # Pygame settings
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size

        self.window = None
        self.clock = None

        # Colors (RGB)
        self.WHITE = (255, 239, 196)
        self.BLACK = (0, 0, 0)
        self.GRAY = (8, 98, 136)
        self.BLUE = (151, 215, 255)
        self.GREEN = (50, 200, 100)
        self.GRID_COLOR = (210, 210, 210)

    def initialize_obstacle(self):
        for col in range(2, 8):
            self.obstacle_positions.add((3, col))

        for col in range(7, 13):
            self.obstacle_positions.add((10, col))

        for row in range(5, 11):
            self.obstacle_positions.add((row, 5))

        for row in range(2, 8):
            self.obstacle_positions.add((row, 11))

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.step_count = 0
        free_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.obstacle_positions:
                    free_cells.append((i, j))
        self.robot_position = free_cells[self.np_random.integers(len(free_cells))]
        self.goal_position = free_cells[self.np_random.integers(len(free_cells))]
        while self.goal_position == self.robot_position or not self.path_exists(self.robot_position, self.goal_position):
            self.goal_position = free_cells[self.np_random.integers(len(free_cells))]
        self.recent_positions.clear()
        self.recent_positions.append(self.robot_position)
        observation = self.get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1
        d_r, d_c = self.actions[action]
        next_position = (self.robot_position[0] + d_r, self.robot_position[1] + d_c)
        is_collision = 1 - self.is_valid_position(next_position)

        if is_collision:
            reward = -20
        else:
            old_distance = self.ManhattanDistance(self.robot_position, self.goal_position)
            self.robot_position = next_position
            new_distance = self.ManhattanDistance(self.robot_position, self.goal_position)
            reward = -1
            if new_distance < old_distance:
                reward += 1
            elif new_distance > old_distance:
                reward -= 1
        self.recent_positions.append(self.robot_position)
        terminated = False

        if self.robot_position == self.goal_position:
            reward = 100
            terminated = True
        truncated = (self.step_count >= self.max_steps) and not terminated
        observation = self.get_observation()
        info = {
            "collision": is_collision,
            "success": terminated
        }
        return observation, reward, terminated, truncated, info

    def ManhattanDistance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_observation(self):
        robot_r, robot_c = self.robot_position
        goal_r, goal_c = self.goal_position
        obstacles = []
        recent_visit = []
        for action in range(4):
            dr, dc = self.actions[action]
            next_position = (robot_r+dr, robot_c+dc)
            blocked = not self.is_valid_position(next_position)
            obstacles.append(int(blocked))
            visit = self.recent_positions.count(next_position)
            recent_visit.append(visit)
        return [robot_r, robot_c, goal_r, goal_c, *obstacles, *recent_visit]

    def is_valid_position(self, position):
        row, col = position
        return (0 <= row < self.grid_size and 0 <= col < self.grid_size and position not in self.obstacle_positions)

    def path_exists(self, start, goal):
        queue = deque()
        visit = set()
        queue.append(start)
        visit.add(start)
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while queue:
            for _ in range(len(queue)):
                curr_pos = queue.popleft()
                if curr_pos == goal:
                    return True
                for dr, dc in directions:
                    new_r, new_c = curr_pos[0] + dr, curr_pos[1] + dc
                    if (new_r, new_c) not in visit and self.is_valid_position((new_r, new_c)):
                        queue.append((new_r, new_c))
                        visit.add((new_r, new_c))
        return False

    def shortest_path_length(self, start, goal):
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            position, distance = queue.popleft()
            if position == goal:
                return distance
            for dr, dc in self.actions.values():
                next_position = (position[0] + dr, position[1] + dc)
                if next_position not in visited and self.is_valid_position(next_position):
                    visited.add(next_position)
                    queue.append((next_position, distance + 1))
        return None

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            pygame.display.set_caption("Robot Navigation")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.WHITE)

        # Draw obstacles
        for row, col in self.obstacle_positions:
            pygame.draw.rect(
                canvas,
                self.GRAY,
                pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
            )

        # Draw goal
        goal_row, goal_col = self.goal_position

        pygame.draw.circle(
            canvas,
            self.GREEN,
            (
                goal_col * self.cell_size + self.cell_size // 2,
                goal_row * self.cell_size + self.cell_size // 2
            ),
            self.cell_size // 3
        )

        # Draw robot
        robot_row, robot_col = self.robot_position

        pygame.draw.circle(
            canvas,
            self.BLUE,
            (
                robot_col * self.cell_size + self.cell_size // 2,
                robot_row * self.cell_size + self.cell_size // 2
            ),
            self.cell_size // 3
        )

        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                self.GRID_COLOR,
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size)
            )

            pygame.draw.line(
                canvas,
                self.GRID_COLOR,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size)
            )

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(canvas).transpose(1, 0, 2)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None