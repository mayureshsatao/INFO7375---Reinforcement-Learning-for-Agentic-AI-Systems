"""
Deep Q-Network (DQN) Agent for Validation Testing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Tuple, List


class DQNNetwork(nn.Module):
    """Deep Q-Network for state-action value estimation"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for learning optimal validation testing strategies"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.lr = config['rl']['learning_rate']
        self.gamma = config['rl']['gamma']
        self.batch_size = config['rl']['batch_size']
        self.buffer_size = config['rl']['buffer_size']
        self.target_update_interval = config['rl']['target_update_interval']

        self.epsilon = config['rl']['exploration_initial_eps']
        self.epsilon_min = config['rl']['exploration_final_eps']
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (
                config['rl']['total_timesteps'] * config['rl']['exploration_fraction']
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.steps = 0
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            action = np.random.uniform(
                low=[0, 0, -10, -10, -10],
                high=[9, 1, 10, 10, 10]
            )
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.cpu().numpy()[0]
                action = np.clip(action, [0, 0, -10, -10, -10], [9, 1, 10, 10, 10])

        return action

    def train_step(self) -> float:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(current_q_values.mean(dim=1), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.losses.append(loss.item())
        return loss.item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']

    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'buffer_size': len(self.replay_buffer)
        }