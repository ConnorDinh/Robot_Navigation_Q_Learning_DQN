import numpy as np
import random

class QLearningAgent:
    def __init__(self, grid_size, learning_rate, gamma, epsilon):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.grid_size = grid_size
        self.action_num = 4
        self.q_table = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.action_num))
        

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_num-1)

        actions = self.q_table[state[0]][state[1]][state[2]][state[3]]
        best_value = np.max(actions)
        best_actions = np.flatnonzero(actions == best_value)
        return int(random.choice(best_actions))

    def update(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state[0]][state[1]][state[2]][state[3]][action]

        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state[0]][next_state[1]][next_state[2]][next_state[3]])

        self.q_table[state[0]][state[1]][state[2]][state[3]][action] += self.learning_rate * (target - current_q)
