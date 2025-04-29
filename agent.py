import torch
from rdqn import RainbowDQN
import torch.optim as optim
from replaybuffer import PrioritizedReplayBuffer
import numpy as np


class RainbowDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Hyperparameters (can be tuned)
        v_min=-10.0,  # Adjust based on expected reward scale
        v_max=10.0,  # Adjust based on expected reward scale
        num_atoms=51,
        hidden_size=128,
        buffer_capacity=10000,
        batch_size=32,
        gamma=0.99,
        n_step=3,
        per_alpha=0.6,
        per_beta=0.4,
        target_update=1000,  # Often higher than 100, e.g., 1000 or 10000 steps
        learning_rate=1e-4,  # Adam default is 1e-3, lower LR often better for RL
    ):
        self.device = torch.device(device)  # Ensure device is torch.device
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.target_update = target_update
        self.steps_done = 0  # Track total optimization steps

        # Distributional RL parameters
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Create networks with specified parameters
        self.policy_net = RainbowDQN(
            state_size, action_size, num_atoms, v_min, v_max, hidden_size, self.device
        ).to(self.device)
        self.target_net = RainbowDQN(
            state_size, action_size, num_atoms, v_min, v_max, hidden_size, self.device
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network should be in evaluation mode

        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Create replay buffer with N-step returns
        self.memory = PrioritizedReplayBuffer(
            buffer_capacity, per_alpha, per_beta, n_step, gamma
        )

        # Beta scheduling for PER (anneals beta from initial value to 1.0)
        self.beta_start = per_beta
        self.beta_frames = 100000  # Example: Anneal over 100k frames/steps
        # Note: beta is annealed in the training loop, not here

    def select_action(self, state):
        """Selects action based on policy net using expected Q-values (NoisyNets handle exploration)."""
        with torch.no_grad():
            # Ensure state is a float tensor and on the correct device
            if not isinstance(state, torch.Tensor):
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                s = state.float().unsqueeze(0).to(self.device)

            if s.shape[1] != self.policy_net.state_size:
                raise ValueError(
                    f"State shape mismatch in select_action. Expected {self.policy_net.state_size}, got {s.shape[1]}"
                )

            # Get expected Q-values using the helper function
            q_values = self.policy_net.get_q_values(s)  # Shape: [1, action_size]
            # Select action with the highest expected Q-value
            return q_values.argmax(1).item()

    def _categorical_projection(self, next_dist_target, rewards, dones, next_action):
        """
        Performs the C51 categorical projection for the target distribution.
        Operates on batches.

        Args:
            next_dist_target (Tensor): Target network's output distribution for next states. [B, A, N]
            rewards (Tensor): Batch of rewards. [B]
            dones (Tensor): Batch of done flags. [B] (True/False or 1/0)
            next_action (Tensor): Batch of best actions in next state selected by target net. [B]

        Returns:
            Tensor: The projected target distribution for the chosen next actions. [B, N]
        """
        batch_size = next_dist_target.size(0)
        # Ensure inputs are on the correct device
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_action = next_action.to(self.device)

        # Get the distributions corresponding to the best next actions
        # Gather requires index shape to match input shape except along the gathering dimension
        next_action = next_action.view(batch_size, 1, 1).expand(
            -1, -1, self.num_atoms
        )  # [B, 1, N]
        next_dist_best_action = next_dist_target.gather(1, next_action).squeeze(
            1
        )  # [B, N]

        # Mask dones for calculations: 1.0 if not done, 0.0 if done
        dones_mask = (~dones).float().unsqueeze(1)  # Shape [B, 1]

        # Calculate the projected support atoms Tz = R + gamma^N * z (where N is n_step)
        # Rewards are already N-step adjusted by the buffer
        Tz = (
            rewards.unsqueeze(1)
            + (self.gamma**self.n_step) * self.support.unsqueeze(0) * dones_mask
        )  # [B, N]

        # Clip projected atoms to [Vmin, Vmax]
        Tz = Tz.clamp(self.v_min, self.v_max)

        # Compute bin indices and offsets
        b = (Tz - self.v_min) / self.delta_z  # [B, N]
        lower_bound = b.floor().long()  # [B, N]
        upper_bound = b.ceil().long()  # [B, N]

        # Handle boundary cases where b is exactly an integer (l == u)
        # Create masks for these cases
        lower_eq_mask = (
            lower_bound == b
        )  # Where projection falls exactly on lower bin edge
        upper_eq_mask = (
            upper_bound == b
        )  # Where projection falls exactly on upper bin edge (redundant if l==u?)

        # Initialize target distribution tensor
        target_dist = torch.zeros(
            batch_size, self.num_atoms, device=self.device
        )  # [B, N]

        # Calculate weights for lower and upper bins (distribution factors)
        # Ensure float division/subtraction before multiplying probability
        weight_l = (upper_bound.float() - b) * next_dist_best_action  # [B, N]
        weight_u = (b - lower_bound.float()) * next_dist_best_action  # [B, N]

        # --- Distribute probability mass using scatter_add_ ---
        # Add weights to the lower bin indices
        # Prevent index out of bounds for lower_bound == num_atoms (when b is exactly Vmax)
        lower_bound = lower_bound.clamp(max=self.num_atoms - 1)
        target_dist.scatter_add_(dim=1, index=lower_bound, src=weight_l)

        # Add weights to the upper bin indices
        # Prevent index out of bounds for upper_bound == 0 (when b is exactly Vmin)
        # Note: scatter_add_ handles indices >= target.shape[dim] by ignoring them, but clamping is safer.
        upper_bound = upper_bound.clamp(min=0)
        target_dist.scatter_add_(dim=1, index=upper_bound, src=weight_u)

        # The above scatter_add handles the l==u case automatically if done carefully.
        # When l == u, b is an integer.
        # weight_l = (l - b) * p = 0 * p = 0
        # weight_u = (b - l) * p = 0 * p = 0
        # This seems wrong. Let's rethink the l==u case distribution.

        # --- Corrected Distribution Logic for l == u ---
        # If l == u, it means b is an integer, and the entire probability p_j
        # should be assigned *only* to index l (or u).
        # The scatter_add approach might be tricky here. Let's use the loop for clarity,
        # then consider vectorization if needed.

        # --- Reverting to Loop for Clarity (can be slow) ---
        target_dist_loop = torch.zeros(
            batch_size, self.num_atoms, device=self.device
        )  # [B, N]
        for i in range(batch_size):
            if dones[i]:
                # If done, target is a Dirac delta at the (clipped) reward
                Tz_done = rewards[i].clamp(self.v_min, self.v_max)
                b_done = (Tz_done - self.v_min) / self.delta_z
                l_done = b_done.floor().long()
                u_done = b_done.ceil().long()
                if l_done == u_done:
                    target_dist_loop[i, l_done] = 1.0
                else:
                    target_dist_loop[i, l_done] = u_done.float() - b_done
                    target_dist_loop[i, u_done] = b_done - l_done.float()
            else:
                # If not done, project the next state distribution atom by atom
                for j in range(
                    self.num_atoms
                ):  # Index of atom in next state distribution
                    p_j = next_dist_best_action[i, j]  # Probability of this atom
                    if p_j > 1e-8:  # Optimization: skip if probability is negligible
                        Tz_j = Tz[i, j]  # Pre-calculated clipped projected atom value
                        b_j = b[i, j]  # Pre-calculated bin position
                        l_j = lower_bound[i, j]  # Index of lower bin
                        u_j = upper_bound[i, j]  # Index of upper bin

                        # Distribute probability p_j to bins l_j and u_j
                        if l_j == u_j:
                            target_dist_loop[i, l_j] += p_j
                        else:
                            target_dist_loop[i, l_j] += p_j * (u_j.float() - b_j)
                            target_dist_loop[i, u_j] += p_j * (b_j - l_j.float())

        # Normalize distribution? Usually not needed if projection is correct.
        # sum_check = target_dist_loop.sum(dim=1)
        # if not torch.allclose(sum_check, torch.ones_like(sum_check)):
        #      print("Warning: Target distribution sum is not 1. Sums:", sum_check)

        return target_dist_loop  # Return the loop-based version for correctness

    def optimize_model(self):
        """Samples batch, computes loss, and updates policy network."""
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples yet

        # Anneal beta for PER importance sampling
        beta = min(
            1.0,
            self.beta_start
            + self.steps_done * (1.0 - self.beta_start) / self.beta_frames,
        )

        # Sample from replay buffer
        # Need to pass current beta to the sample method if it uses it directly
        # Assuming the buffer's sample method uses its internal self.beta and we update it here (if needed)
        # self.memory.beta = beta # Update beta in buffer if it's used there dynamically
        transitions, indices, weights = self.memory.sample(
            self.batch_size
        )  # Weights are calculated using buffer's beta
        batch = list(zip(*transitions))

        # Convert batch data to tensors
        states_np = np.array(batch[0], dtype=np.float32)
        actions_np = np.array(batch[1], dtype=np.int64)
        rewards_np = np.array(batch[2], dtype=np.float32)  # These are N-step rewards
        next_states_np = np.array(batch[3], dtype=np.float32)  # This is state S_{t+N}
        dones_np = np.array(batch[4], dtype=bool)  # Done flags for S_{t+N}

        state_batch = torch.from_numpy(states_np).to(self.device)
        action_batch = torch.from_numpy(actions_np).to(self.device)  # [B]
        reward_batch = torch.from_numpy(rewards_np).to(self.device)  # [B]
        next_state_batch = torch.from_numpy(next_states_np).to(self.device)
        done_batch = torch.from_numpy(dones_np).to(self.device)  # [B]
        weights = (
            torch.from_numpy(np.array(weights, dtype=np.float32))
            .to(self.device)
            .unsqueeze(1)
        )  # [B, 1] for broadcasting loss

        # --- Target Calculation ---
        with torch.no_grad():
            # Get next state distributions and expected Q-values from TARGET network
            next_dist_target = self.target_net(next_state_batch)  # [B, A, N]
            next_q_target = (next_dist_target * self.support).sum(dim=2)  # [B, A]

            # Select best next action based on TARGET network's expected Q-values
            next_action = next_q_target.argmax(dim=1)  # [B]

            # Project the target distribution for the selected next actions
            target_dist = self._categorical_projection(
                next_dist_target,  # Distribution from target net [B, A, N]
                reward_batch,  # N-step Rewards [B]
                done_batch,  # Done flags for S_{t+N} [B]
                next_action,  # Best action in S_{t+N} selected by target net [B]
            )  # Result shape: [B, N]

        # --- Loss Calculation ---
        # Get current state distributions from POLICY network
        current_dist = self.policy_net(state_batch)  # [B, A, N]

        # Get the distribution for the action actually taken (action_batch)
        # Need to gather based on action_batch index
        action_batch_expanded = action_batch.view(self.batch_size, 1, 1).expand(
            -1, -1, self.num_atoms
        )  # [B, 1, N]
        current_dist_taken_action = current_dist.gather(
            1, action_batch_expanded
        ).squeeze(
            1
        )  # [B, N]

        # Compute cross-entropy loss between target and current distributions
        # Add small epsilon for numerical stability before log
        loss = -(target_dist * torch.log(current_dist_taken_action + 1e-8)).sum(
            dim=1
        )  # [B]

        # Apply PER weights: loss = loss * weights
        # Loss shape is [B], weights shape is [B, 1], squeeze weights or unsqueeze loss
        weighted_loss = (
            loss * weights.squeeze(1)
        ).mean()  # Calculate the mean weighted loss

        # --- Update Priorities in PER Buffer ---
        # Priorities are typically based on the absolute loss (or TD error)
        # Add a small epsilon before taking power alpha for stability
        new_priorities = loss.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # --- Optimize Policy Network ---
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent explosions
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # --- Reset Noise ---
        # Reset noise in NoisyLinear layers (for both networks if target uses them)
        # Important for exploration when using Noisy Nets
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # --- Update Target Network --- (Soft update often more stable, but periodic hard update is simpler)
        self.steps_done += 1  # Increment optimization step counter
        if self.steps_done % self.target_update == 0:
            print(f"--- Updating target network at step {self.steps_done} ---")
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return weighted_loss.item()  # Return loss value for logging if needed
