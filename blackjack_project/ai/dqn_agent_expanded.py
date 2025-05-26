import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim=1, output_dim=2):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, input_dim=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_play = DQN(input_dim, 2).to(self.device)
        self.target_model_play = DQN(input_dim, 2).to(self.device)
        self.optimizer_play = optim.Adam(self.model_play.parameters(), lr=0.001)

        self.model_bet = DQN(input_dim, 3).to(self.device)
        self.target_model_bet = DQN(input_dim, 3).to(self.device)
        self.optimizer_bet = optim.Adam(self.model_bet.parameters(), lr=0.001)

        self.criterion = nn.MSELoss()
        self.epsilon = 0.1
        self.gamma = 0.99
        self.batch_size = 32
        self.memory_play = deque(maxlen=10000)
        self.memory_bet = deque(maxlen=10000)

    def choose_action(self, state, phase='PLAY'):
        if isinstance(state, dict):
            state = [state['value']]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if phase == 'BETTING':
            if random.random() < self.epsilon:
                return random.choice([50, 100, 200])
            with torch.no_grad():
                q_values = self.model_bet(state)
                action = q_values.argmax().item()
                return [50, 100, 200][action]
        else:
            if random.random() < self.epsilon:
                return random.choice([0, 1])
            with torch.no_grad():
                q_values = self.model_play(state)
                return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done, phase='PLAY'):
        if isinstance(state, dict):
            state = [state['value']]
        if isinstance(next_state, dict):
            next_state = [next_state['value']]

        if phase == 'BETTING':
            action_index = [50, 100, 200].index(action)
            self.memory_bet.append((state, action_index, reward, next_state, done))
        else:
            self.memory_play.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory_play) >= self.batch_size:
            batch = random.sample(self.memory_play, self.batch_size)
            self._train_on_batch(batch, self.model_play, self.target_model_play, self.optimizer_play)

        if len(self.memory_bet) >= self.batch_size:
            batch = random.sample(self.memory_bet, self.batch_size)
            self._train_on_batch(batch, self.model_bet, self.target_model_bet, self.optimizer_bet)

    def _train_on_batch(self, batch, model, target_model, optimizer):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_values = model(states).gather(1, actions)
        next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + self.gamma * next_q_values * (~dones)

        loss = self.criterion(q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_target_model(self):
        self.target_model_play.load_state_dict(self.model_play.state_dict())
        self.target_model_bet.load_state_dict(self.model_bet.state_dict())
