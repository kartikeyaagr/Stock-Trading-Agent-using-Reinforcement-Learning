import torch
from dqn import RainbowDQN, PrioritizedReplayBuffer
import torch.optim as optim
import numpy as np


class RainbowDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        v_min=-10.0,
        v_max=10.0,
        num_atoms=51,
        hidden_size=128,
        buffer_capacity=100000,
        batch_size=32,
        gamma=0.99,
        n_step=3,
        per_alpha=0.6,
        per_beta=0.4,
        target_update=1000,
        learning_rate=1e-4,
        beta_annealing_steps=100000,
    ):
        self.device = torch.device(device)
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.target_update = target_update
        self.steps_done = 0
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.policy_net = RainbowDQN(
            state_size, action_size, num_atoms, v_min, v_max, hidden_size, self.device
        ).to(self.device)
        self.target_net = RainbowDQN(
            state_size, action_size, num_atoms, v_min, v_max, hidden_size, self.device
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(
            buffer_capacity, per_alpha, per_beta, n_step, gamma
        )
        self.beta_start = per_beta
        self.beta_frames = beta_annealing_steps

    def select_action(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                s = state.float().unsqueeze(0).to(self.device)
            if s.shape[1] != self.policy_net.state_size:
                raise ValueError(
                    f"State shape mismatch in select_action. Expected {self.policy_net.state_size}, got {s.shape[1]}"
                )
            q_values = self.policy_net.get_q_values(s)
            return q_values.argmax(1).item()

    def _categorical_projection(
        self,
        next_dist_target: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = next_dist_target.size(0)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_action = next_action.to(self.device)
        rewards = rewards.view(batch_size, 1)
        dones_mask = (~dones).float().view(batch_size, 1)
        next_action_expanded = next_action.view(batch_size, 1, 1).expand(
            -1, -1, self.num_atoms
        )
        next_dist_best_action = next_dist_target.gather(
            1, next_action_expanded
        ).squeeze(1)
        Tz = rewards + (self.gamma**self.n_step) * self.support.view(1, -1) * dones_mask
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        lower_bound = b.floor().long()
        upper_bound = b.ceil().long()
        target_dist = torch.zeros(batch_size, self.num_atoms, device=self.device)
        weight_l = (upper_bound.float() - b) * next_dist_best_action
        weight_u = (b - lower_bound.float()) * next_dist_best_action
        target_dist.scatter_add_(1, lower_bound.clamp(0, self.num_atoms - 1), weight_l)
        target_dist.scatter_add_(1, upper_bound.clamp(0, self.num_atoms - 1), weight_u)
        done_indices = torch.where(dones)[0]
        num_done = len(done_indices)
        if num_done > 0:
            Tz_done = rewards[done_indices].clamp(self.v_min, self.v_max)
            b_done = (Tz_done - self.v_min) / self.delta_z
            l_done = b_done.floor().long()
            u_done = b_done.ceil().long()
            target_dist_done = torch.zeros(num_done, self.num_atoms, device=self.device)
            weight_ld = (u_done.float() - b_done).squeeze(1)
            weight_ud = (b_done - l_done.float()).squeeze(1)
            eq_mask = (l_done == u_done).squeeze(1)
            weight_ld[eq_mask] = 0.0
            weight_ud[eq_mask] = 0.0
            target_dist_done.scatter_(
                1, l_done.clamp(0, self.num_atoms - 1), weight_ld.unsqueeze(1)
            )
            target_dist_done.scatter_add_(
                1, u_done.clamp(0, self.num_atoms - 1), weight_ud.unsqueeze(1)
            )
            if torch.any(eq_mask):
                l_done_eq = l_done[eq_mask].clamp(0, self.num_atoms - 1)
                src_ones = torch.ones(l_done_eq.size(0), device=self.device).unsqueeze(
                    1
                )
                target_dist_done.scatter_(1, l_done_eq, src_ones)
            target_dist[done_indices] = target_dist_done
        return target_dist

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        beta = min(
            1.0,
            self.beta_start
            + self.steps_done * (1.0 - self.beta_start) / self.beta_frames,
        )
        transitions, indices, weights_np = self.memory.sample(
            self.batch_size, beta=beta
        )
        if transitions is None:
            return None

        states, actions, rewards, next_states, dones = zip(*transitions)
        state_batch = torch.from_numpy(np.array(states, dtype=np.float32)).to(
            self.device
        )
        action_batch = torch.from_numpy(np.array(actions, dtype=np.int64)).to(
            self.device
        )
        reward_batch = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(
            self.device
        )
        next_state_batch = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(
            self.device
        )
        done_batch = torch.from_numpy(np.array(dones, dtype=bool)).to(self.device)
        weights = torch.from_numpy(weights_np).float().to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_dist_target = self.target_net(next_state_batch)
            next_q_target = (next_dist_target * self.support).sum(dim=2)
            next_action = next_q_target.argmax(dim=1)
            target_dist = self._categorical_projection(
                next_dist_target,
                reward_batch,
                done_batch,
                next_action,
            )

        current_dist = self.policy_net(state_batch)
        action_batch_expanded = action_batch.view(self.batch_size, 1, 1).expand(
            -1, -1, self.num_atoms
        )
        current_dist_taken_action = current_dist.gather(
            1, action_batch_expanded
        ).squeeze(1)
        loss = -(target_dist * torch.log(current_dist_taken_action + 1e-8)).sum(dim=1)
        weighted_loss = (loss * weights.squeeze(1)).mean()
        new_priorities = loss.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            print(f"--- Updating target network at step {self.steps_done} ---")
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return weighted_loss.item()
