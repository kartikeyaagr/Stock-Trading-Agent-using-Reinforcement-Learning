from collections import deque
import numpy as np


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
        """Calculate multi-step return, next state, and done."""
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = n_s, d

        state, action = self.n_step_buffer[0][:2]
        return state, action, reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        """Add experience to the n-step buffer and main buffer."""
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
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
