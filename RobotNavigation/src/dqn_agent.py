import torch
import torch.nn as nn
from collections import deque
import random
from torch.optim import Adam

class QNetwork(nn.Module):
    def __init__(self, state_dim=12, action_dim=4):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 128)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_dim)

    def forward(self,state):
        x = self.layer_1(state)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, is_done):
        self.buffer.append((state, action, reward, next_state, is_done))

    def sample(self, batch_size):
        random_samples = random.sample(self.buffer, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        is_dones = []
        for state, action, reward, next_state, is_done in random_samples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            is_dones.append(is_done)

        return states, actions, rewards, next_states, is_dones

    def length(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, double_dqn=False):
        self.policy_network = QNetwork(12, 4)
        self.target_network = QNetwork(12, 4)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = Adam(self.policy_network.parameters(), lr=0.0005)
        self.buffer = ReplayBuffer(100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.batch_size = 64
        self.update_frequency = 1000
        self.double_dqn = double_dqn
        self.history_length = 5

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state = self.preprocess_state(state)
        with torch.no_grad():
            q_values = self.policy_network.forward(state)
        return int(torch.argmax(q_values).item())

    def preprocess_state(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).clone()
        state[..., :4] /= 14.0
        state[..., 8:12] /= self.history_length
        return state

    def update(self):
        if self.buffer.length() < self.batch_size:
            return
        states, actions, rewards, next_states, is_dones = self.buffer.sample(self.batch_size)
        s = self.preprocess_state(states)
        a = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32)
        n = self.preprocess_state(next_states)
        i = torch.tensor(is_dones, dtype=torch.float32)
        predicted_q = self.policy_network.forward(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_network.forward(n).argmax(dim=1, keepdim=True)
                next_q = self.target_network.forward(n).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_network.forward(n).max(dim=1).values
            targets = r + self.gamma * (1 - i) * next_q
        loss_fn = nn.HuberLoss()
        loss = loss_fn(predicted_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()
