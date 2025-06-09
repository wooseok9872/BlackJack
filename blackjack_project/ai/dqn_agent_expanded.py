import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size=5):
        self.state_size = state_size
        self.memory_play = deque(maxlen=20000)
        self.memory_bet = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0            # ✅ 초기 탐험률 복원
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_play = self._build_play_model()
        self.target_model_play = self._build_play_model()
        self.optimizer_play = optim.Adam(self.model_play.parameters(), lr=self.learning_rate)

        self.model_bet = self._build_bet_model()
        self.target_model_bet = self._build_bet_model()
        self.optimizer_bet = optim.Adam(self.model_bet.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss()

        if os.path.exists("ai_dqn_trained.pt"):
            self.load("ai_dqn_trained.pt")

    def _build_play_model(self):
        return DQN(self.state_size, 2).to(self.device)  # HIT, STAND

    def _build_bet_model(self):
        return DQN(self.state_size, 3).to(self.device)  # BET: 50, 100, 200

    def _get_tensor_from_state(self, state_dict):
        value = state_dict.get('player_value', 0) / 21.0
        dealer_value = state_dict.get('dealer_value', 0) / 11.0
        usable_ace = float(state_dict.get('usable_ace', 0))
        is_soft_hand = float(state_dict.get('is_soft_hand', 0))
        bust_risk = float(state_dict.get('bust_risk', 0))

        state_tensor = torch.FloatTensor([
            value,
            dealer_value,
            usable_ace,
            is_soft_hand,
            bust_risk
        ]).unsqueeze(0).to(self.device)

        return state_tensor

    def choose_action(self, state, phase="PLAY", chips=700, episode=None):
        state_tensor = self._get_tensor_from_state(state)
        if np.random.rand() <= self.epsilon:
            if phase == "BETTING":
                valid_bets = [a for a in [50, 100, 200] if a <= chips]
                return random.choice(valid_bets) if valid_bets else 0
            else:
                return random.choice([0, 1])
        else:
            with torch.no_grad():
                if phase == "BETTING":
                    q_values = self.model_bet(state_tensor)
                    valid_indices = [i for i, bet in enumerate([50, 100, 200]) if bet <= chips]
                    if not valid_indices:
                        print(f"[BETTING FAIL] No valid bet for chips={chips}")
                        return 0
                    filtered_q = q_values[0][valid_indices]
                    best_idx = valid_indices[torch.argmax(filtered_q).item()]
                    selected_bet = [50, 100, 200][best_idx]
                    if episode is not None and episode % 1000 == 0:
                        print(f"[BET] EP {episode} | Chips: {chips} | Qs: {q_values.cpu().numpy().flatten()} | Selected: {selected_bet}")
                    return selected_bet
                else:
                    q_values = self.model_play(state_tensor)
                    player_val = state.get("player_value", 0)
                    dealer_val = state.get("dealer_value", 0)

                    if episode is not None and episode % 1000 == 0 and player_val <= 16:
                        q_hit = q_values[0][0].item()
                        q_stand = q_values[0][1].item()
                        print(f"[Q] [EP {episode}] AI: {player_val} vs Dealer: {dealer_val} | Q(HIT): {q_hit:.3f}, Q(STAND): {q_stand:.3f}")

                    return torch.argmax(q_values[0]).item()

    def remember(self, state, action, reward, next_state, done, phase='PLAY'):
        state_tensor = self._get_tensor_from_state(state).squeeze(0).cpu().numpy()
        next_state_tensor = self._get_tensor_from_state(next_state).squeeze(0).cpu().numpy()

        if phase == 'BETTING':
            action_index = [50, 100, 200].index(action)
            self.memory_bet.append((state_tensor, action_index, reward, next_state_tensor, done))
        else:
            self.memory_play.append((state_tensor, action, reward, next_state_tensor, done))

    def replay(self):
        if len(self.memory_play) >= self.batch_size:
            batch = random.sample(self.memory_play, self.batch_size)
            self._train_on_batch(batch, self.model_play, self.target_model_play, self.optimizer_play)

        if len(self.memory_bet) >= self.batch_size:
            batch = random.sample(self.memory_bet, self.batch_size)
            self._train_on_batch(batch, self.model_bet, self.target_model_bet, self.optimizer_bet)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

    def save(self, path):
        torch.save({
            'play_model': self.model_play.state_dict(),
            'bet_model': self.model_bet.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model_play.load_state_dict(checkpoint['play_model'])
        self.model_bet.load_state_dict(checkpoint['bet_model'])
