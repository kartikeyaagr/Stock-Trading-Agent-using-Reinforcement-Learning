from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import wandb  # Import W&B


def collect_stock_data(path):
    print("Loading pre-downloaded stock data...")
    DATA_FILE_PATH = path
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
    df["Returns"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_middle"] = rolling_mean
    df["BB_upper"] = rolling_mean + (2 * rolling_std)
    df["BB_lower"] = rolling_mean - (2 * rolling_std)
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    return df


class TradingEnvironment:
    def __init__(
        self, data: pd.DataFrame, initial_balance=100000, config=None
    ):  # Added config
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
        self.config = config if config else {}  # Store config

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
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0
        self.portfolio_value = self.initial_balance
        self.returns_history = []
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        current_data = self.data.iloc[self.current_step]
        current_price = current_data[self.price_column]
        position_value = self.position * current_price
        normalized_position_value = (
            position_value / self.portfolio_value
            if self.portfolio_value > 1e-6
            else 0.0
        )
        normalized_balance = self.balance / self.initial_balance
        normalized_portfolio_value = self.portfolio_value / self.initial_balance
        agent_state = [
            normalized_position_value,
            normalized_balance,
            normalized_portfolio_value,
        ]
        tech_state = []
        epsilon = 1e-8
        safe_price = current_price if abs(current_price) > epsilon else epsilon
        for col in self.feature_columns:
            value = current_data[col]
            if pd.isna(value):
                value = 0.0
            if col == "Returns":
                scaled_value = value
            elif col in [
                "SMA_20",
                "SMA_50",
                "BB_upper",
                "BB_lower",
                "MACD",
                "Signal_Line",
            ]:
                scaled_value = (value - current_price) / safe_price
            elif col == "RSI":
                scaled_value = value / 100.0
            elif col == "Volatility":
                scaled_value = value
            else:
                scaled_value = value
            tech_state.append(scaled_value)
        state = agent_state + tech_state
        state_np = np.array(state, dtype=np.float32)
        if np.any(np.isnan(state_np)) or np.any(np.isinf(state_np)):
            state_np = np.nan_to_num(state_np, nan=0.0, posinf=0.0, neginf=0.0)
        if len(state_np) != self.state_size:
            raise RuntimeError(
                f"Internal Error: State size mismatch. Expected {self.state_size}, got {len(state_np)}"
            )
        return state_np

    def step(self, action):
        if self.current_step >= len(self.data) - 2:
            current_state = self._get_state()
            return current_state, 0.0, True
        current_data = self.data.iloc[self.current_step]
        next_data = self.data.iloc[self.current_step + 1]
        price = current_data[self.price_column]
        next_price = next_data[self.price_column]
        if pd.isna(price):
            reward = 0.0
            done = self.current_step >= len(self.data) - 2
            self.current_step += 1
            return self._get_state(), reward, done
        if pd.isna(next_price):
            next_price = price
        initial_portfolio_value = self.portfolio_value
        if action == 1:
            if self.position == 0 and self.balance > price:
                shares_to_buy = self.balance // price
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    self.balance -= cost
                    self.position = shares_to_buy
        elif action == 2:
            if self.position > 0:
                revenue = self.position * price
                self.balance += revenue
                self.position = 0
        portfolio_value = self.balance + self.position * next_price
        returns = (
            (portfolio_value - initial_portfolio_value) / initial_portfolio_value
            if initial_portfolio_value > 1e-6
            else 0.0
        )
        self.returns_history.append(returns)
        self.portfolio_value = portfolio_value
        self.current_step += 1
        reward = returns

        # CVaR reward adjustment from config
        cvar_alpha = self.config.get("cvar_alpha", 0.05)
        min_history_for_cvar = self.config.get("min_history_for_cvar", 20)
        cvar_penalty_factor = self.config.get("cvar_penalty_factor", 0.1)

        if (
            len(self.returns_history) >= min_history_for_cvar
            and cvar_penalty_factor > 0
        ):
            calculated_cvar = self._calculate_cvar(
                self.returns_history[-min_history_for_cvar:], alpha=cvar_alpha
            )
            reward = returns - cvar_penalty_factor * abs(calculated_cvar)

        done = self.current_step >= len(self.data) - 1
        next_state = self._get_state()
        return next_state, reward, done

    def _calculate_cvar(self, returns, alpha=0.05):
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)
        if len(returns) == 0:
            return 0.0
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else 0.0


class NoisyLinear(nn.Module):
    def __init__(
        self, in_features, out_features, std_init=0.5
    ):  # std_init can come from config
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
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.weight_epsilon.device != x.device:
            self.weight_epsilon = self.weight_epsilon.to(x.device)
            self.bias_epsilon = self.bias_epsilon.to(x.device)
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        num_atoms,
        v_min,
        v_max,
        hidden_size,
        noisy_std_init,
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
            NoisyLinear(hidden_size, hidden_size, std_init=noisy_std_init),
            nn.ReLU(),
        ).to(self.device)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2, std_init=noisy_std_init),
            nn.ReLU(),
            NoisyLinear(hidden_size // 2, num_atoms, std_init=noisy_std_init),
        ).to(self.device)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size // 2, std_init=noisy_std_init),
            nn.ReLU(),
            NoisyLinear(
                hidden_size // 2, action_size * num_atoms, std_init=noisy_std_init
            ),
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
        state, action, n_step_reward, next_state, done = (
            self._get_n_step_info()
        )  # n_step_reward is key
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, n_step_reward, next_state, done))
        else:
            self.buffer[self.position] = (
                state,
                action,
                n_step_reward,
                next_state,
                done,
            )
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


class RainbowDQNAgent:
    def __init__(self, state_size, action_size, device, config):  # Pass full config
        self.device = torch.device(device)
        self.action_size = action_size
        self.config = config  # Store config
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.n_step = config["n_step"]
        self.target_update = config["target_update_steps"]
        self.steps_done = 0
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]
        self.num_atoms = config["num_atoms"]
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(
            self.device
        )
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.policy_net = RainbowDQN(
            state_size,
            action_size,
            self.num_atoms,
            self.v_min,
            self.v_max,
            config["hidden_size"],
            config["noisy_std_init"],
            self.device,
        ).to(self.device)
        self.target_net = RainbowDQN(
            state_size,
            action_size,
            self.num_atoms,
            self.v_min,
            self.v_max,
            config["hidden_size"],
            config["noisy_std_init"],
            self.device,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config["learning_rate"]
        )
        self.memory = PrioritizedReplayBuffer(
            config["buffer_capacity"],
            config["per_alpha"],
            config["per_beta_start"],
            self.n_step,
            self.gamma,
        )
        self.beta_start = config["per_beta_start"]
        self.beta_frames = config["per_beta_frames"]
        self.n_step_returns_log = []  # For logging N-step returns

    def select_action(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                s = state.float().unsqueeze(0).to(self.device)
            if s.shape[1] != self.policy_net.state_size:
                raise ValueError(
                    f"State shape mismatch. Expected {self.policy_net.state_size}, got {s.shape[1]}"
                )
            q_values = self.policy_net.get_q_values(s)
            return q_values.argmax(1).item()

    def _categorical_projection(
        self,
        next_dist_target_full: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_actions_policy: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = next_dist_target_full.size(0)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_actions_policy = next_actions_policy.to(
            self.device
        )  # Actions from policy net

        # Gather the distributions from target_net corresponding to actions chosen by policy_net
        next_action_expanded = next_actions_policy.view(batch_size, 1, 1).expand(
            -1, -1, self.num_atoms
        )
        next_dist_best_action = next_dist_target_full.gather(
            1, next_action_expanded
        ).squeeze(1)

        dones_mask = (~dones).float().view(batch_size, 1)
        Tz = (
            rewards.view(batch_size, 1)
            + (self.gamma**self.n_step) * self.support.view(1, -1) * dones_mask
        )
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
        self.memory.beta = (
            beta  # Update beta in buffer for sampling if it uses it directly
        )
        transitions, indices, weights_np = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        states_np = np.array(batch[0], dtype=np.float32)
        actions_np = np.array(batch[1], dtype=np.int64)
        rewards_np = np.array(batch[2], dtype=np.float32)
        next_states_np = np.array(batch[3], dtype=np.float32)
        dones_np = np.array(batch[4], dtype=bool)

        # Collect N-step returns for logging
        self.n_step_returns_log.extend(rewards_np.tolist())

        state_batch = torch.from_numpy(states_np).to(self.device)
        action_batch = torch.from_numpy(actions_np).to(self.device)
        reward_batch = torch.from_numpy(rewards_np).to(self.device)
        next_state_batch = torch.from_numpy(next_states_np).to(self.device)
        done_batch = torch.from_numpy(dones_np).to(self.device)
        weights = (
            torch.from_numpy(np.array(weights_np, dtype=np.float32))
            .to(self.device)
            .unsqueeze(1)
        )

        with torch.no_grad():
            # --- Double DQN Part ---
            # 1. Select best action for next state using policy_net's Q-values
            next_q_policy = self.policy_net.get_q_values(
                next_state_batch
            )  # Shape: [batch_size, action_size]
            next_actions_policy = next_q_policy.argmax(dim=1)  # Shape: [batch_size]

            # 2. Get the distribution for this 'next_action' from the target_net
            next_dist_target_full = self.target_net(
                next_state_batch
            )  # Shape: [batch_size, action_size, num_atoms]

            target_dist = self._categorical_projection(
                next_dist_target_full, reward_batch, done_batch, next_actions_policy
            )

        current_dist_policy = self.policy_net(state_batch)
        action_batch_expanded = action_batch.view(self.batch_size, 1, 1).expand(
            -1, -1, self.num_atoms
        )
        current_dist_taken_action = current_dist_policy.gather(
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

        # Log intermediate values
        if (
            self.steps_done % self.config.get("log_intermediate_freq_steps", 100) == 0
        ):  # Log every N optimization steps
            q_values_policy_all_actions = (current_dist_policy * self.support).sum(
                dim=2
            )
            q_values_taken_action_policy = q_values_policy_all_actions.gather(
                1, action_batch.unsqueeze(1)
            ).squeeze(1)
            avg_q_value_policy = q_values_taken_action_policy.mean().item()
            avg_td_error = np.mean(new_priorities - 1e-6)  # Subtract epsilon

            wandb.log(
                {
                    "avg_q_value_policy_taken": avg_q_value_policy,
                    "avg_td_error": avg_td_error,
                    "beta_per": beta,  # Log current beta for PER
                    "mean_loss_raw": loss.mean().item(),  # Raw unweighted loss
                },
                step=self.steps_done,
            )

        if self.steps_done % self.target_update == 0:
            print(f"--- Updating target network at step {self.steps_done} ---")
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return weighted_loss.item()


def train_agent(env, agent, num_episodes, max_steps_per_episode, config):  # Pass config
    returns_history = []
    losses_history = []
    initial_balance = config["initial_balance"]

    for episode in range(num_episodes):
        state = env.reset()
        episode_losses = []
        done = False
        steps_in_episode = 0
        while not done and steps_in_episode < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            loss_val = agent.optimize_model()
            if loss_val is not None:
                episode_losses.append(loss_val)
            steps_in_episode += 1

        returns_history.append(env.portfolio_value)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_history.append(avg_loss)

        print(
            f"Episode {episode + 1}/{num_episodes}, Steps: {steps_in_episode}, "
            f"Total Steps: {agent.steps_done}, Portfolio: {env.portfolio_value:.2f}, Avg Loss: {avg_loss:.4f}"
        )

        wandb.log(
            {
                "episode": episode + 1,
                "portfolio_value_eoe": env.portfolio_value,
                "return_pct_eoe": (
                    (env.portfolio_value - initial_balance) / initial_balance
                )
                * 100,
                "avg_loss_eoe": avg_loss,
                "total_agent_steps": agent.steps_done,
                "episode_steps": steps_in_episode,
            },
            step=agent.steps_done,
        )  # Log with agent steps as primary x-axis

        # Log N-step returns statistics periodically
        if (
            episode % config.get("log_n_step_return_freq_episodes", 10) == 0
            and agent.n_step_returns_log
        ):
            n_step_array = np.array(agent.n_step_returns_log)
            print(
                f"\nN-step Return Stats (last {len(agent.n_step_returns_log)} optimization samples):"
            )
            print(
                f"  Min: {n_step_array.min():.4f}, Max: {n_step_array.max():.4f}, Mean: {n_step_array.mean():.4f}"
            )
            print(
                f"  Percentiles: 5th={np.percentile(n_step_array, 5):.4f}, 95th={np.percentile(n_step_array, 95):.4f}"
            )
            wandb.log(
                {"n_step_returns_hist": wandb.Histogram(n_step_array)},
                step=agent.steps_done,
            )
            agent.n_step_returns_log = []  # Reset for next interval

    return returns_history, losses_history, agent


def simulate_agent_on_data(
    agent: RainbowDQNAgent, eval_data: pd.DataFrame, initial_balance: float
) -> tuple[list, list]:
    print(f"Simulating agent on evaluation data...")
    eval_env = TradingEnvironment(eval_data, initial_balance=initial_balance)
    state = eval_env.reset()
    done = False
    portfolio_values = [initial_balance]
    actions_taken = []

    agent.policy_net.eval()

    while not done:
        with torch.no_grad():
            action = agent.select_action(state)

        next_state, reward, done = eval_env.step(action)

        portfolio_values.append(eval_env.portfolio_value)
        actions_taken.append(action)
        state = next_state

    agent.policy_net.train()
    print(
        f"Agent simulation complete. Final portfolio value: {portfolio_values[-1]:.2f}"
    )
    return portfolio_values, actions_taken


def simulate_buy_and_hold(
    eval_data: pd.DataFrame, initial_balance: float
) -> tuple[list, float]:
    print("Simulating Buy and Hold strategy...")
    if eval_data.empty:
        print("Warning: Cannot simulate Buy and Hold on empty data.")
        return [initial_balance], 0.0

    start_price = eval_data["Close"].iloc[0]
    end_price = eval_data["Close"].iloc[-1]

    num_shares = initial_balance // start_price
    cash_left = initial_balance - (num_shares * start_price)

    daily_values = (eval_data["Close"] * num_shares) + cash_left
    portfolio_values = [initial_balance] + daily_values.tolist()

    final_value = portfolio_values[-1]
    total_return_pct = ((final_value - initial_balance) / initial_balance) * 100

    print(
        f"Buy & Hold simulation complete. Final portfolio value: {final_value:.2f}, Total Return: {total_return_pct:.2f}%"
    )
    return portfolio_values, total_return_pct


def calculate_performance_metrics(
    portfolio_values: list, risk_free_rate: float = 0.0
) -> dict:
    metrics = {"Sharpe Ratio": 0.0, "Max Drawdown": 0.0}
    if len(portfolio_values) < 2:
        print("Warning: Need at least 2 portfolio values to calculate metrics.")
        return metrics

    values_series = pd.Series(portfolio_values)
    daily_returns = values_series.pct_change().dropna()

    if not daily_returns.empty and daily_returns.std() != 0:
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std()
        metrics["Sharpe Ratio"] = sharpe * np.sqrt(252)

    cumulative_max = values_series.cummax()
    drawdown = ((values_series - cumulative_max) / cumulative_max) * 100
    metrics["Max Drawdown"] = drawdown.min() if not drawdown.empty else 0.0

    return metrics


def calculate_drawdown_series(portfolio_values: list) -> pd.Series:
    portfolio_values_series = pd.Series(portfolio_values)
    if len(portfolio_values_series) < 2:
        return pd.Series(dtype=float)
    cumulative_max = portfolio_values_series.cummax()
    drawdown = (
        (portfolio_values_series - cumulative_max) / cumulative_max.replace(0, 1e-9)
    ) * 100
    return drawdown.fillna(0)


def plot_portfolio_value(
    agent_portfolio_values: list,
    bnh_portfolio_values: list,
    initial_balance: float,
    eval_data_index: pd.DatetimeIndex,
    save_path: str,
):
    print(f"Generating Portfolio Value plot...")
    y_agent = agent_portfolio_values[1:]
    num_sim_steps = len(y_agent)
    if num_sim_steps == 0:
        print("Skipping portfolio value plot: No agent simulation steps.")
        return
    x_index = eval_data_index[:num_sim_steps]
    y_bnh_full = bnh_portfolio_values[1:]
    y_bnh_aligned = (
        y_bnh_full[:num_sim_steps]
        if len(y_bnh_full) >= num_sim_steps
        else y_bnh_full + [y_bnh_full[-1]] * (num_sim_steps - len(y_bnh_full))
    )  # Pad if shorter
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_index, y_agent, label="Agent Portfolio", color="tab:blue", linewidth=1.5)
    ax.plot(
        x_index,
        y_bnh_aligned,
        label="Buy & Hold Portfolio",
        color="tab:orange",
        linewidth=1.5,
    )
    ax.axhline(y=initial_balance, color="gray", linestyle="--", label="Initial Balance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Agent vs. Buy & Hold Portfolio Value (Test Set)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.ticklabel_format(style="plain", axis="y")
    fig.autofmt_xdate()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Portfolio Value plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving Portfolio Value plot: {e}")
    plt.close(fig)
    if wandb.run:
        wandb.log({"eval_portfolio_value_plot": wandb.Image(save_path)})


def plot_drawdown(
    agent_portfolio_values: list,
    bnh_portfolio_values: list,
    eval_data_index: pd.DatetimeIndex,
    save_path: str,
):
    print(f"Generating Drawdown plot...")
    agent_drawdown_full = calculate_drawdown_series(agent_portfolio_values)
    bnh_drawdown_full = calculate_drawdown_series(bnh_portfolio_values)
    y_agent_dd = agent_drawdown_full[1:]
    num_sim_steps = len(y_agent_dd)
    if num_sim_steps == 0:
        print("Skipping drawdown plot: No agent simulation steps.")
        return
    x_index = eval_data_index[:num_sim_steps]
    y_bnh_dd_full = bnh_drawdown_full[1:]
    y_bnh_dd_aligned = (
        y_bnh_dd_full[:num_sim_steps]
        if len(y_bnh_dd_full) >= num_sim_steps
        else y_bnh_dd_full + [y_bnh_dd_full[-1]] * (num_sim_steps - len(y_bnh_dd_full))
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_index, y_agent_dd, label="Agent Drawdown", color="tab:red", linewidth=1.5)
    ax.plot(
        x_index,
        y_bnh_dd_aligned,
        label="Buy & Hold Drawdown",
        color="tab:purple",
        linewidth=1.5,
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Agent vs. Buy & Hold Drawdown Over Time (Test Set)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.autofmt_xdate()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Drawdown plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving Drawdown plot: {e}")
    plt.close(fig)
    if wandb.run:
        wandb.log({"eval_drawdown_plot": wandb.Image(save_path)})


def plot_actions(eval_data_df: pd.DataFrame, agent_actions: list, save_path: str):
    print(f"Generating Actions plot...")
    num_actions = len(agent_actions)
    if eval_data_df.empty or num_actions == 0:
        print("Skipping actions plot.")
        return
    price_data_for_actions = eval_data_df["Close"].iloc[:num_actions]
    dates_for_actions = eval_data_df.index[:num_actions]
    if len(price_data_for_actions) != num_actions:
        print("Warning: Price data length mismatch for actions plot.")
        return
    actions_array = np.array(agent_actions)
    buy_indices = np.where(actions_array == 1)[0]
    sell_indices = np.where(actions_array == 2)[0]
    buy_signal_dates = dates_for_actions[buy_indices]
    sell_signal_dates = dates_for_actions[sell_indices]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        dates_for_actions,
        price_data_for_actions,
        label="Close Price",
        color="black",
        alpha=0.8,
        linewidth=1,
    )
    if len(buy_signal_dates) > 0:
        ax.plot(
            buy_signal_dates,
            price_data_for_actions[buy_indices],
            "^",
            markersize=8,
            color="green",
            label="Buy Signal",
            alpha=0.9,
            linestyle="None",
        )
    if len(sell_signal_dates) > 0:
        ax.plot(
            sell_signal_dates,
            price_data_for_actions[sell_indices],
            "v",
            markersize=8,
            color="red",
            label="Sell Signal",
            alpha=0.9,
            linestyle="None",
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Agent Trading Actions on Price (Test Set)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.autofmt_xdate()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Actions plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving Actions plot: {e}")
    plt.close(fig)
    if wandb.run:
        wandb.log({"eval_actions_plot": wandb.Image(save_path)})


def plot_returns_histogram(
    agent_portfolio_values: list,
    bnh_portfolio_values: list,
    save_path: str,
    bins: int = 50,
):
    print(f"Generating Returns Histogram plot...")
    if len(agent_portfolio_values) <= 1 or len(bnh_portfolio_values) <= 1:
        print("Skipping returns histogram plot.")
        return
    agent_daily_returns = pd.Series(agent_portfolio_values).pct_change().dropna() * 100
    bnh_daily_returns = pd.Series(bnh_portfolio_values).pct_change().dropna() * 100
    if agent_daily_returns.empty and bnh_daily_returns.empty:
        print("No daily returns data to plot histogram.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    if not agent_daily_returns.empty:
        ax.hist(
            agent_daily_returns,
            bins=bins,
            alpha=0.7,
            label="Agent Daily Returns",
            color="tab:blue",
            density=True,
        )
    if not bnh_daily_returns.empty:
        ax.hist(
            bnh_daily_returns,
            bins=bins,
            alpha=0.7,
            label="Buy & Hold Daily Returns",
            color="tab:orange",
            density=True,
        )
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Daily Returns (Test Set)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    if not agent_daily_returns.empty or not bnh_daily_returns.empty:
        ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Returns Histogram plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving Returns Histogram plot: {e}")
    plt.close(fig)
    if wandb.run:
        wandb.log({"eval_returns_hist_plot": wandb.Image(save_path)})


def print_training_charts(returns_history, losses_history, initial_balance, save_path):
    print("Generating training performance plot...")
    if not returns_history or not losses_history:
        print("Skipping training plot.")
        return
    cumulative_returns_pct = [
        ((final_value - initial_balance) / initial_balance) * 100
        for final_value in returns_history
    ]
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_return = "tab:red"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Return (%)", color=color_return)
    ax1.plot(
        cumulative_returns_pct,
        color=color_return,
        label="Cumulative Return %",
        linewidth=1.5,
    )
    ax1.tick_params(axis="y", labelcolor=color_return)
    ax1.axhline(y=0, color="gray", linestyle="--", label="0% Return")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2 = ax1.twinx()
    color_loss = "tab:blue"
    ax2.set_ylabel("Average Loss", color=color_loss)
    ax2.plot(
        losses_history, color=color_loss, alpha=0.7, label="Avg Loss", linewidth=1.5
    )
    ax2.tick_params(axis="y", labelcolor=color_loss)
    ax2.legend(loc="upper right")
    plt.title("Training Performance: Cumulative Return and Average Loss per Episode")
    fig.tight_layout()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training plot saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving training plot: {e}")
    plt.close(fig)
    if wandb.run:
        wandb.log({"training_performance_plot": wandb.Image(save_path)})


def run_evaluation(
    agent: RainbowDQNAgent,
    eval_data_path: str,
    initial_balance: float,
    output_dir: str,
    config,
):  # Pass config
    print("\n" + "=" * 30 + " STARTING EVALUATION " + "=" * 30)
    try:
        eval_data_raw = pd.read_csv(eval_data_path, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Eval data not found at {eval_data_path}. Skipped.")
        return
    except Exception as e:
        print(f"Error loading eval data: {e}. Skipped.")
        return
    processed_eval_data = calculate_technical_indicators(eval_data_raw.copy())
    processed_eval_data.dropna(inplace=True)
    if processed_eval_data.empty:
        print("Error: Eval data empty after preprocessing. Skipped.")
        return
    print(
        f"Eval data range: {processed_eval_data.index.min()} to {processed_eval_data.index.max()}, Shape: {processed_eval_data.shape}"
    )
    os.makedirs(output_dir, exist_ok=True)

    agent_portfolio_values, agent_actions = simulate_agent_on_data(
        agent, processed_eval_data, initial_balance
    )
    bnh_portfolio_values, bnh_total_return_pct = simulate_buy_and_hold(
        processed_eval_data, initial_balance
    )

    agent_final_value = agent_portfolio_values[-1]
    agent_total_return_pct = (
        ((agent_final_value - initial_balance) / initial_balance) * 100
        if initial_balance > 1e-9
        else 0.0
    )
    agent_metrics = calculate_performance_metrics(
        agent_portfolio_values, config.get("risk_free_rate", 0.0)
    )
    bnh_metrics = calculate_performance_metrics(
        bnh_portfolio_values, config.get("risk_free_rate", 0.0)
    )

    print("\n--- Evaluation Results ---")
    print(f"{'Metric':<20} | {'Agent':<15} | {'Buy & Hold':<15}")
    print("-" * 55)
    print(
        f"{'Final Portfolio':<20} | {agent_final_value:<15.2f} | {bnh_portfolio_values[-1]:<15.2f}"
    )
    print(
        f"{'Total Return (%)':<20} | {agent_total_return_pct:<15.2f} | {bnh_total_return_pct:<15.2f}"
    )
    print(
        f"{'Sharpe Ratio':<20} | {agent_metrics['Sharpe Ratio']:<15.2f} | {bnh_metrics['Sharpe Ratio']:<15.2f}"
    )
    print(
        f"{'Max Drawdown (%)':<20} | {agent_metrics['Max Drawdown']:<15.2f} | {bnh_metrics['Max Drawdown']:<15.2f}"
    )
    print("-" * 55)

    if wandb.run:
        wandb.log(
            {
                "eval_agent_final_portfolio": agent_final_value,
                "eval_bnh_final_portfolio": bnh_portfolio_values[-1],
                "eval_agent_total_return_pct": agent_total_return_pct,
                "eval_bnh_total_return_pct": bnh_total_return_pct,
                "eval_agent_sharpe_ratio": agent_metrics["Sharpe Ratio"],
                "eval_bnh_sharpe_ratio": bnh_metrics["Sharpe Ratio"],
                "eval_agent_max_drawdown": agent_metrics["Max Drawdown"],
                "eval_bnh_max_drawdown": bnh_metrics["Max Drawdown"],
            }
        )

    print("\n--- Generating Evaluation Plots ---")
    eval_data_original_index = processed_eval_data.index
    plot_portfolio_value(
        agent_portfolio_values,
        bnh_portfolio_values,
        initial_balance,
        eval_data_original_index,
        os.path.join(output_dir, "evaluation_portfolio_value.png"),
    )
    plot_drawdown(
        agent_portfolio_values,
        bnh_portfolio_values,
        eval_data_original_index,
        os.path.join(output_dir, "evaluation_drawdown.png"),
    )
    plot_actions(
        processed_eval_data,
        agent_actions,
        os.path.join(output_dir, "evaluation_actions.png"),
    )
    plot_returns_histogram(
        agent_portfolio_values,
        bnh_portfolio_values,
        os.path.join(output_dir, "evaluation_returns_histogram.png"),
    )
    print("=" * 30 + " EVALUATION FINISHED " + "=" * 30 + "\n")


if __name__ == "__main__":
    config = {
        "project_name": "rl-stock-trader-rainbow-dqn",
        "run_name": f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        "seed": 42,
        # Paths
        "train_data_path": "/home/kartikeya.agrawal_ug25/RL_Final/data/train_data.csv",
        "eval_data_path": "/home/kartikeya.agrawal_ug25/RL_Final/data/eval_data.csv",
        "output_dir": "/home/kartikeya.agrawal_ug25/RL_Final/output",
        # Training hparams
        "num_train_episodes": 10,  # Reduced for quick testing, original 100
        "max_steps_per_episode": 10000,
        "initial_balance": 100000,
        # Agent hparams
        "v_min": -10.0,  # Adjusted based on typical returns (e.g. -1 to 1, or -10 to 10 if N-step sums are larger)
        "v_max": 10.0,
        "num_atoms": 51,  # Original 101, common 51
        "hidden_size": 128,
        "noisy_std_init": 0.5,
        "buffer_capacity": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "n_step": 3,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_frames": 100000,  # Anneal beta over this many steps
        "target_update_steps": 1000,  # Target network update frequency (steps)
        "learning_rate": 1e-4,  # Common starting point, adjust based on loss behavior
        # Environment hparams
        "cvar_alpha": 0.05,
        "cvar_penalty_factor": 0.0,  # Set to 0 to disable CVaR penalty initially, or small like 0.01, 0.1
        "min_history_for_cvar": 20,
        # Logging hparams
        "log_intermediate_freq_steps": 100,  # How often to log Q-values/TD errors
        "log_n_step_return_freq_episodes": 10,  # How often to log N-step return histogram
        # Evaluation
        "risk_free_rate": 0.0,
    }

    # --- Seeding ---
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            False  # Can slow down, but good for reproducibility
        )
    # --- Initialize W&B ---
    wandb.init(project=config["project_name"], name=config["run_name"], config=config)

    TRAIN_DATA_PATH = config["train_data_path"]
    EVAL_DATA_PATH = config["eval_data_path"]
    OUTPUT_DIR = config["output_dir"]
    NUM_TRAIN_EPISODES = config["num_train_episodes"]
    INITIAL_BALANCE = config["initial_balance"]
    MAX_STEPS_PER_EPISODE = config["max_steps_per_episode"]

    train_stock_data = collect_stock_data(TRAIN_DATA_PATH)
    processed_train_data = calculate_technical_indicators(train_stock_data.copy())
    processed_train_data.dropna(inplace=True)
    print("Training Dataset shape:", processed_train_data.shape)
    print(
        "Training Date range:",
        processed_train_data.index.min(),
        "to",
        processed_train_data.index.max(),
    )

    train_env = TradingEnvironment(
        processed_train_data, initial_balance=INITIAL_BALANCE, config=config
    )
    state_size = train_env.state_size
    action_size = 3
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")
    agent = RainbowDQNAgent(
        state_size, action_size, device=device_name, config=config
    )  # Pass config

    print(f"Starting training for {NUM_TRAIN_EPISODES} episodes...")
    returns_history, losses_history, agent = train_agent(
        train_env,
        agent,
        num_episodes=NUM_TRAIN_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        config=config,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_plot_save_path = os.path.join(
        OUTPUT_DIR, f"training_performance_{NUM_TRAIN_EPISODES}.png"
    )
    print_training_charts(
        returns_history, losses_history, INITIAL_BALANCE, training_plot_save_path
    )

    run_evaluation(
        agent, EVAL_DATA_PATH, INITIAL_BALANCE, OUTPUT_DIR, config=config
    )  # Pass config

    wandb.finish()  # End W&B run
    print("Script finished.")
