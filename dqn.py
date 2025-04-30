import torch
import torch.nn as nn
from noise import NoisyLinear
import numpy as np
from collections import deque


class RainbowDQN(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        num_atoms=51,
        v_min=-10,
        v_max=10,
        hidden_size=128,
        device=None,
    ):
        super(RainbowDQN, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RainbowDQN on device: {self.device}")
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
        ).to(self.device)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, num_atoms),
        ).to(self.device)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, action_size * num_atoms),
        ).to(self.device)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        batch_size = x.size(0)
        features = self.feature_layer(x)
        value_logits = self.value_stream(features)
        advantage_logits = self.advantage_stream(features)
        value_logits = value_logits.view(batch_size, 1, self.num_atoms)
        advantage_logits = advantage_logits.view(
            batch_size, self.action_size, self.num_atoms
        )
        mean_advantage_logits = advantage_logits.mean(dim=1, keepdim=True)
        q_logits = value_logits + (advantage_logits - mean_advantage_logits)
        dist = torch.softmax(q_logits, dim=2)
        return dist

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        if self.support.device != self.device:
            self.support = self.support.to(self.device)
        dist = self.forward(state)
        q_values = (dist * self.support).sum(dim=2)
        return q_values


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []
        self.n_step_buffer = deque(maxlen=n_step)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = n_s, d
        state, action = self.n_step_buffer[0][:2]
        return state, action, reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, next_state, done = self._get_n_step_info()
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        probs = self.priorities[: len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
