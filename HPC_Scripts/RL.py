from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


def collect_stock_data():
    print("Loading pre-downloaded stock data...")
    DATA_FILE_PATH = "reliance_data.csv"
    try:
        stock_data = pd.read_csv(DATA_FILE_PATH, index_col="Date", parse_dates=True)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        print("Please run the data download script first.")
        exit()
    except Exception as e:
        print(f"Error loading data from file: {e}")
        exit()
    return stock_data


def calculate_technical_indicators(df):
    # Calculate returns
    df["Returns"] = df["Close"].pct_change()

    # Calculate moving averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Calculate RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Calculate MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_middle"] = rolling_mean
    df["BB_upper"] = rolling_mean + (2 * rolling_std)
    df["BB_lower"] = rolling_mean - (2 * rolling_std)

    # Calculate volatility
    df["Volatility"] = df["Returns"].rolling(window=20).std()

    return df


class TradingEnvironment:
    def __init__(self, data: pd.DataFrame, initial_balance=100000):

        if data.isnull().values.any():

            nan_rows = data[data.isnull().any(axis=1)]
            print(
                "Warning: DataFrame passed to TradingEnvironment contains NaN values."
            )
            print("Ensure .dropna() was called after calculating indicators.")
            print("First few rows with NaNs:\n", nan_rows.head())

            raise ValueError(
                "DataFrame contains NaNs. Please clean before passing to Environment."
            )

        self.data = data.copy()
        self.initial_balance = initial_balance

        self.feature_columns = [
            "Returns",
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "Signal_Line",
            "BB_upper",
            "BB_lower",
            "Volatility",
        ]
        self.price_column = "Close"

        # --- Verify all required columns are present ---
        required_cols_for_state = self.feature_columns + [self.price_column]
        missing_cols = [
            col for col in required_cols_for_state if col not in self.data.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")

        self.state_size = 3 + len(self.feature_columns)
        print(f"Environment initialized. State size: {self.state_size}")

        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.balance = self.initial_balance
        self.current_step = 0  # Start from the first row (assuming NaNs are dropped)
        self.position = 0  # Number of shares held
        self.portfolio_value = self.initial_balance
        self.returns_history = []
        # print(f"Environment reset. Starting step: {self.current_step}, Index: {self.data.index[self.current_step]}")
        return self._get_state()

    def _get_state(self):

        if self.current_step >= len(self.data):
            # Should not happen if step logic is correct, but safeguard
            print(
                f"Warning: _get_state called at step {self.current_step} >= data length {len(self.data)}. Using last valid data."
            )
            self.current_step = len(self.data) - 1

        current_data = self.data.iloc[self.current_step]
        current_price = current_data[self.price_column]

        # --- 1. Agent Status Features (Normalized) ---
        position_value = self.position * current_price
        normalized_position_value = (
            position_value / self.portfolio_value
            if self.portfolio_value > 1e-6
            else 0.0
        )  # Avoid div by zero
        normalized_balance = self.balance / self.initial_balance
        normalized_portfolio_value = self.portfolio_value / self.initial_balance

        agent_state = [
            normalized_position_value,
            normalized_balance,
            normalized_portfolio_value,
        ]

        # --- 2. Technical Indicator Features (Scaled/Normalized) ---
        tech_state = []
        # Use a small epsilon to prevent division by zero if price is exactly zero
        epsilon = 1e-8
        safe_price = current_price if abs(current_price) > epsilon else epsilon

        # Iterate through defined feature columns and apply scaling/normalization
        for col in self.feature_columns:
            value = current_data[col]
            if pd.isna(value):
                # This shouldn't happen if data is pre-cleaned, but as a fallback
                print(
                    f"Warning: NaN found in column '{col}' at step {self.current_step}. Replacing with 0."
                )
                value = 0.0

            # Apply scaling/normalization based on the indicator type
            if col == "Returns":
                # Returns are often small, maybe scale slightly? Or keep as is. Let's keep as is for now.
                scaled_value = value
            elif col in [
                "SMA_20",
                "SMA_50",
                "BB_upper",
                "BB_lower",
                "MACD",
                "Signal_Line",
            ]:
                # Normalize price-based indicators relative to the current price
                scaled_value = (value - current_price) / safe_price
            elif col == "RSI":
                # Scale RSI from [0, 100] to [0.0, 1.0]
                scaled_value = value / 100.0
            elif col == "Volatility":
                # Volatility is a std dev (percentage), usually small. Keep as is for now.
                scaled_value = value
            else:
                # Default for any unexpected columns (shouldn't happen with check in init)
                scaled_value = value

            tech_state.append(scaled_value)

        # --- Combine states ---
        state = agent_state + tech_state

        # Final check for NaNs or Infs that might have slipped through calculations
        state_np = np.array(state, dtype=np.float32)
        if np.any(np.isnan(state_np)) or np.any(np.isinf(state_np)):
            # print(f"Warning: NaN/Inf detected in final state at step {self.current_step}. Replacing with 0.")
            # print(f"Original state: {state}")
            state_np = np.nan_to_num(
                state_np, nan=0.0, posinf=0.0, neginf=0.0
            )  # Replace NaN/Inf with 0

        # Ensure state size matches expectation
        if len(state_np) != self.state_size:
            raise RuntimeError(
                f"Internal Error: State size mismatch. Expected {self.state_size}, got {len(state_np)}"
            )

        return state_np

    # --- Keep the step, _calculate_cvar methods as they were in the previous fix ---
    # (Make sure they use self.price_column instead of hardcoded "Close")
    def step(self, action):
        # Actions: 0 = hold, 1 = buy, 2 = sell

        if self.current_step >= len(self.data) - 2:  # Need current and next price
            # print(f"Attempting to step beyond data bounds (step {self.current_step}). Ending episode.")
            current_state = self._get_state()  # Get the last valid state
            return current_state, 0.0, True  # Return last state, 0 reward, done=True

        current_data = self.data.iloc[self.current_step]
        next_data = self.data.iloc[self.current_step + 1]

        price = current_data[self.price_column]
        next_price = next_data[self.price_column]

        # Handle potential NaN prices more robustly
        if pd.isna(price):
            # Price is NaN: Cannot trade reliably. Treat as forced hold, zero reward.
            # print(f"Warning: NaN price encountered at step {self.current_step}. Forcing hold.")
            reward = 0.0
            done = self.current_step >= len(self.data) - 2  # Check done condition again
            self.current_step += 1  # Move step forward
            # Portfolio value likely shouldn't change if price is NaN
            # self.portfolio_value remains the same
            return self._get_state(), reward, done
        if pd.isna(next_price):
            # Next price is NaN: Can execute trade at current price, but reward/next value is uncertain.
            # Option: use current price for next value calculation (conservative)
            # print(f"Warning: NaN next_price encountered at step {self.current_step+1}. Using current price for value calculation.")
            next_price = price  # Fallback

        initial_portfolio_value = self.portfolio_value

        if action == 1:  # Buy
            if (
                self.position == 0 and self.balance > price
            ):  # Check if enough balance for at least 1 share
                shares_to_buy = self.balance // price
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    self.balance -= cost
                    self.position = shares_to_buy
            # If already holding or not enough balance, treat as hold

        elif action == 2:  # Sell
            if self.position > 0:
                revenue = self.position * price
                self.balance += revenue
                self.position = 0
            # If not holding, treat as hold

        # Update portfolio value using next_price (potentially the fallback price)
        portfolio_value = self.balance + self.position * next_price

        # Calculate returns based on the change from initial value for this step
        if initial_portfolio_value > 1e-6:  # Avoid division by zero
            returns = (
                portfolio_value - initial_portfolio_value
            ) / initial_portfolio_value
        else:
            returns = 0.0

        self.returns_history.append(returns)
        self.portfolio_value = (
            portfolio_value  # Update portfolio value tracked by the env
        )

        # Advance step *before* getting the next state
        self.current_step += 1

        # CVaR reward adjustment
        reward = returns
        alpha = 0.05
        min_history_for_cvar = 20
        if len(self.returns_history) >= min_history_for_cvar:
            cvar_penalty_factor = 0.1
            calculated_cvar = self._calculate_cvar(
                self.returns_history[-min_history_for_cvar:], alpha=alpha
            )
            reward = returns - cvar_penalty_factor * abs(calculated_cvar)

        # Check if done (reached the end of data)
        # Now done condition is simpler: if current_step points beyond the last valid index
        done = (
            self.current_step >= len(self.data) - 1
        )  # -1 because step was already incremented

        next_state = self._get_state()  # Get state for the *new* current_step

        return next_state, reward, done

    def _calculate_cvar(self, returns, alpha=0.05):
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)
        if len(returns) == 0:
            return 0.0
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else 0.0


"""## 4. Rainbow DQN Implementation

Now, let's implement the Rainbow DQN with all its components.
"""


# Assuming NoisyLinear class is defined as before:
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(
            size, device=self.weight_mu.device
        )  # Ensure noise is on correct device
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        # Ensure noise tensors are on the same device as parameters/input
        if self.weight_epsilon.device != x.device:
            self.weight_epsilon = self.weight_epsilon.to(x.device)
            self.bias_epsilon = self.bias_epsilon.to(x.device)
            # print(f"Moved noise to {x.device} in NoisyLinear") # Optional debug print

        if self.training:
            # Sample new noise only if training
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean weights/biases during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


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


"""## 5. Training Loop

Finally, let's implement the training loop for our Rainbow DQN agent.
"""


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
        buffer_capacity=100000,
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


# --- Rest of the script (train_agent, __main__, etc.) remains the same ---
# ... (make sure train_agent calls agent.optimize_model() and handles the return)
# ... (make sure __main__ increments agent.steps_done *after* optimize_model if not done inside)
# Note: agent.steps_done is incremented inside optimize_model now. The training loop
# doesn't need to increment it separately.


# Example modification in train_agent loop:
def train_agent(
    env, agent, num_episodes=1000, max_steps_per_episode=10000
):  # Add max steps
    returns_history = []
    losses_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_losses = []
        done = False
        steps_in_episode = 0  # Track steps within episode

        while not done and steps_in_episode < max_steps_per_episode:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done = env.step(action)

            # Store transition in memory (using N-step buffer)
            agent.memory.push(state, action, reward, next_state, done)

            # Move to next state
            state = next_state
            episode_return += reward

            # Optimize model (only if buffer is large enough)
            loss_val = agent.optimize_model()  # optimize_model increments steps_done
            if loss_val is not None:
                episode_losses.append(loss_val)

            steps_in_episode += 1

        returns_history.append(
            env.portfolio_value
        )  # Store final portfolio value or total return
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_history.append(avg_loss)
        print(
            f"Episode {episode + 1}/{num_episodes}, "
            f"Steps: {steps_in_episode}, "
            f"Total Steps: {agent.steps_done}, "
            f"Return: {env.portfolio_value:.2f}, "  # Print final portfolio value
            f"Avg Loss: {avg_loss:.4f}"
        )

    return returns_history, losses_history, agent  # Return losses too


def evaluate_agent(
    agent: RainbowDQNAgent, eval_data: pd.DataFrame, initial_balance: float
) -> tuple[list, list]:
    """
    Runs the trained agent on unseen evaluation data without learning.

    Args:
        agent (RainbowDQNAgent): The trained agent instance.
        eval_data (pd.DataFrame): The evaluation dataset (e.g., test set).
        initial_balance (float): The starting balance for evaluation.

    Returns:
        tuple[list, list]:
            - portfolio_values (list): List of portfolio values at each step.
            - actions_taken (list): List of actions taken at each step.
    """
    print(
        f"Evaluating agent on data from {eval_data.index.min()} to {eval_data.index.max()}"
    )
    eval_env = TradingEnvironment(eval_data, initial_balance=initial_balance)
    state = eval_env.reset()
    done = False

    portfolio_values = [initial_balance]  # Start with initial balance
    actions_taken = []

    # Set agent to evaluation mode (disables noise in NoisyLinear, etc.)
    agent.policy_net.eval()

    while not done:
        with torch.no_grad():  # Ensure no gradient calculations
            action = agent.select_action(state)  # Use the learned policy

        next_state, reward, done = eval_env.step(action)  # Environment dynamics
        portfolio_values.append(eval_env.portfolio_value)  # Record value AFTER step
        actions_taken.append(action)
        state = next_state

    # Set agent back to training mode if needed elsewhere
    agent.policy_net.train()
    print(f"Evaluation complete. Final portfolio value: {portfolio_values[-1]:.2f}")
    return portfolio_values, actions_taken


def calculate_sharpe_ratio(
    portfolio_values: list, risk_free_rate: float = 0.0
) -> float:
    """Calculates the annualized Sharpe Ratio."""
    portfolio_values_series = pd.Series(portfolio_values)
    daily_returns = portfolio_values_series.pct_change().dropna()

    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0  # Cannot calculate Sharpe ratio if no returns or zero std dev

    excess_returns = daily_returns - risk_free_rate / 252  # Daily risk-free rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    annualized_sharpe = sharpe_ratio * np.sqrt(
        252
    )  # Annualize assuming 252 trading days

    return annualized_sharpe


def calculate_max_drawdown(portfolio_values: list) -> float:
    """Calculates the Maximum Drawdown (MDD)."""
    portfolio_values_series = pd.Series(portfolio_values)
    if portfolio_values_series.empty:
        return 0.0

    cumulative_max = portfolio_values_series.cummax()
    drawdown = (portfolio_values_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()  # Most negative value is the MDD

    return max_drawdown * 100  # Return as percentage


def calculate_buy_and_hold(
    eval_data: pd.DataFrame, initial_balance: float
) -> tuple[list, float, float]:
    """Calculates Buy-and-Hold performance and daily values for MDD."""
    if eval_data.empty:
        return [initial_balance], 0.0, 0.0

    start_price = eval_data["Close"].iloc[0]
    end_price = eval_data["Close"].iloc[-1]
    num_shares = initial_balance // start_price
    cash_left = initial_balance - (num_shares * start_price)

    # Calculate daily portfolio values for Buy & Hold
    bnh_portfolio_values = [initial_balance]  # Start
    daily_values = (eval_data["Close"] * num_shares) + cash_left
    bnh_portfolio_values.extend(daily_values.tolist())  # Add daily values

    final_value = (num_shares * end_price) + cash_left
    total_return = (final_value - initial_balance) / initial_balance * 100

    print(f"\nBuy & Hold Results:")
    print(f"  Initial Value: {initial_balance:.2f}")
    print(f"  Shares Bought: {num_shares} @ {start_price:.2f}")
    print(f"  Final Value: {final_value:.2f}")
    print(f"  Total Return: {total_return:.2f}%")

    bnh_mdd = calculate_max_drawdown(bnh_portfolio_values)
    print(f"  Max Drawdown: {bnh_mdd:.2f}%")

    return bnh_portfolio_values, total_return, bnh_mdd


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Collect Data
    stock_data = collect_stock_data()
    processed_data = calculate_technical_indicators(stock_data.copy())
    processed_data.dropna(inplace=True)

    # EDA Output
    print("Dataset shape:", processed_data.shape)
    print("  Date range:", processed_data.index.min(), "to", processed_data.index.max())
    # print(processed_data.head()) # Keep short

    # Create environment and agent
    env = TradingEnvironment(processed_data, initial_balance=100000)  # Example balance
    state_size = env.state_size
    action_size = 3  # Buy, Hold, Sell

    # Use a specific device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")

    agent = RainbowDQNAgent(
        state_size, action_size, device=device_name
    )  # Pass explicit device

    # --- Checkpoint Loading ---
    load_checkpoint = False  # Set to True to load
    load_path = "/home/kartikeya.agrawal_ug25/RL_Final/rainbow_dqn_agent_checkpoint_new.pth"  # Example path

    if load_checkpoint:
        try:
            # Map location ensures model loads correctly even if saved on GPU and loading on CPU
            checkpoint = torch.load(load_path, map_location=agent.device)
            agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            agent.steps_done = checkpoint.get(
                "steps_done", 0
            )  # Use get for backward compatibility
            # Consider loading replay buffer state if saved
            print(
                f"Checkpoint successfully loaded from {load_path}. Resuming at step {agent.steps_done}."
            )
            # Ensure target net is in eval mode after loading
            agent.target_net.eval()
        except FileNotFoundError:
            print(
                f"Checkpoint file not found at {load_path}. Starting training from scratch."
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")

    # Train the agent
    num_train_episodes = 40  # Adjust number of episodes
    print(f"Starting training for {num_train_episodes} episodes...")
    returns_history, losses_history, agent = train_agent(
        env, agent, num_episodes=num_train_episodes
    )

    # Plot returns (final portfolio value) and losses
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = "tab:red"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Final Portfolio Value", color=color)
    ax1.plot(returns_history, color=color, label="Portfolio Value")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axhline(
        y=env.initial_balance, color="gray", linestyle="--", label="Initial Balance"
    )
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:blue"
    ax2.set_ylabel("Average Loss", color=color)
    ax2.plot(losses_history, color=color, alpha=0.6, label="Avg Loss")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Training Performance: Portfolio Value and Average Loss per Episode")
    plt.savefig(
        "/home/kartikeya.agrawal_ug25/RL_Final/output/plot.png",
        dpi=300,
        transparent=True,
    )

    # --- Checkpoint Saving ---
    save_path = "/home/kartikeya.agrawal_ug25/RL_Final/output/rainbow_dqn_agent_checkpoint_final.pth"  # Example path
    try:
        torch.save(
            {
                "policy_net_state_dict": agent.policy_net.state_dict(),
                "target_net_state_dict": agent.target_net.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "steps_done": agent.steps_done,
                # Optional: Save Vmin/Vmax/NumAtoms if they might change
                "v_min": agent.v_min,
                "v_max": agent.v_max,
                "num_atoms": agent.num_atoms,
                # Optional: Save replay buffer (can be large)
                # "replay_buffer": agent.memory
            },
            save_path,
        )
        print(f"Checkpoint saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

    # ---------------------------------
    #         Evaluate Agent
    # ---------------------------------

    initial_balance = 100000
    eval_data = pd.read_csv("eval_data.csv")
    processed_eval_data = calculate_technical_indicators(eval_data.copy())
    processed_eval_data.dropna(inplace=True)
    agent_portfolio_values, _ = evaluate_agent(
        agent, processed_eval_data, initial_balance
    )

    agent_final_value = agent_portfolio_values[-1]
    agent_total_return = (agent_final_value - initial_balance) / initial_balance * 100
    agent_sharpe = calculate_sharpe_ratio(agent_portfolio_values)
    agent_mdd = calculate_max_drawdown(agent_portfolio_values)

    print(f"\nRainbow DQN Agent Test Results:")
    print(f"  Final Portfolio Value: {agent_final_value:.2f}")
    print(f"  Total Return: {agent_total_return:.2f}%")
    print(f"  Annualized Sharpe Ratio: {agent_sharpe:.2f}")
    print(f"  Maximum Drawdown: {agent_mdd:.2f}%")

    # --- Buy and Hold Baseline ---
    bnh_portfolio_values, bnh_total_return, bnh_mdd = calculate_buy_and_hold(
        stock_data, initial_balance
    )
    # Sharpe for BnH isn't directly comparable if just using total return,
    # but you could calculate it from its daily returns if needed:
    bnh_sharpe = calculate_sharpe_ratio(bnh_portfolio_values)
    print(f"  (BnH Annualized Sharpe Ratio: {bnh_sharpe:.2f})")  # For comparison

    print("\n--- Comparison Table ---")
    print("| Metric                     | Rainbow DQN Agent | Buy-and-Hold Baseline |")
    print("| :------------------------- | :---------------- | :-------------------- |")
    print(
        f"| Total Return               | {agent_total_return: >17.2f}% | {bnh_total_return: >21.2f}% |"
    )
    print(
        f"| Annualized Sharpe Ratio    | {agent_sharpe: >17.2f} | {bnh_sharpe: >21.2f} |"
    )
    print(f"| Maximum Drawdown (MDD)     | {agent_mdd: >17.2f}% | {bnh_mdd: >21.2f}% |")
    print(
        f"| Initial Balance            | {initial_balance: >17,.0f} | {initial_balance: >21,.0f} |"
    )
    print(
        f"| Final Portfolio Value      | {agent_final_value: >17,.2f} | {bnh_portfolio_values[-1]: >21,.2f} |"
    )
