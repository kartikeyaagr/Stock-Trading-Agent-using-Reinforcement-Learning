import numpy as np
import torch
import torch.nn as nn
from noisylinear import NoisyLinear


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
        """
        Rainbow DQN Network with Dueling Architecture.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
            num_atoms (int): Number of atoms for the distributional RL value distribution.
            v_min (float): Minimum value for the distribution support.
            v_max (float): Maximum value for the distribution support.
            hidden_size (int): Size of the hidden layers.
            device (torch.device): Device to run the network on (CPU or CUDA).
        """
        super(RainbowDQN, self).__init__()

        # device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RainbowDQN on device: {self.device}")

        # save sizes
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # support for C51 (distributional)
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)

        # --- Shared Feature Extraction Layers ---
        # Using NoisyLinear for exploration baked into the network
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),  # First layer can be standard Linear
            nn.ReLU(),
            NoisyLinear(hidden_size, hidden_size),  # Subsequent layers are noisy
            nn.ReLU(),
        ).to(
            self.device
        )  # Ensure layers are moved to the correct device

        # --- Dueling Architecture Streams ---
        # 1. Value Stream: Estimates V(s) - output shape [batch_size, num_atoms]
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),  # Smaller layer for value
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, num_atoms),
        ).to(self.device)

        # 2. Advantage Stream: Estimates A(s, a) - output shape [batch_size, action_size * num_atoms]
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2),  # Smaller layer for advantage
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, action_size * num_atoms),
        ).to(self.device)

        # Move the entire module to the specified device AFTER initializing layers
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Dueling Rainbow DQN network.

        Args:
            x (torch.Tensor): Input state tensor, shape [batch_size, state_size].

        Returns:
            torch.Tensor: Output distribution over atoms for each action's Q-value,
                          shape [batch_size, action_size, num_atoms].
        """
        # Ensure input tensor is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            # print(f"Moved input tensor to {self.device} in forward pass") # Optional debug print

        batch_size = x.size(0)

        # 1. Pass through shared feature layer
        features = self.feature_layer(x)  # Shape: [batch_size, hidden_size]

        # 2. Pass features through value and advantage streams
        value_logits = self.value_stream(features)  # Shape: [batch_size, num_atoms]
        advantage_logits = self.advantage_stream(
            features
        )  # Shape: [batch_size, action_size * num_atoms]

        # 3. Reshape streams for combination
        # Reshape value to be broadcastable: [batch_size, 1, num_atoms]
        value_logits = value_logits.view(batch_size, 1, self.num_atoms)
        # Reshape advantage: [batch_size, action_size, num_atoms]
        advantage_logits = advantage_logits.view(
            batch_size, self.action_size, self.num_atoms
        )

        # 4. Combine Value and Advantage streams (Dueling formula applied to logits)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, .)))
        # Calculate mean advantage across actions for each atom
        mean_advantage_logits = advantage_logits.mean(
            dim=1, keepdim=True
        )  # Shape: [batch_size, 1, num_atoms]

        # Combine using broadcasting
        q_logits = value_logits + (
            advantage_logits - mean_advantage_logits
        )  # Shape: [batch_size, action_size, num_atoms]

        # 5. Apply Softmax to get the probability distribution over atoms for each action
        # Softmax is applied along the last dimension (atoms)
        dist = torch.softmax(
            q_logits, dim=2
        )  # Shape: [batch_size, action_size, num_atoms]

        # Optional: Add a small epsilon to prevent log(0) issues later if needed,
        # although the cross-entropy loss usually handles this.
        # dist = dist.clamp(min=1e-8)

        return dist

    def reset_noise(self):
        """Resets the noise in all NoisyLinear layers."""
        # Iterate all child modules recursively
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the expected Q-values for each action from the output distribution.

        Args:
            state (torch.Tensor): Input state tensor, shape [batch_size, state_size].

        Returns:
            torch.Tensor: Expected Q-values for each action, shape [batch_size, action_size].
        """
        # Ensure support is on the correct device
        if self.support.device != self.device:
            self.support = self.support.to(self.device)

        dist = self.forward(state)  # Get the distribution [batch, action, atoms]
        # Calculate expected value: sum(probability * support_value) for each action
        q_values = (dist * self.support).sum(dim=2)  # Shape: [batch, action]
        return q_values
