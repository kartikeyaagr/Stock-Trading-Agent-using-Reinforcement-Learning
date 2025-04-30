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
    DATA_FILE_PATH = "/home/kartikeya.agrawal_ug25/RL_Final/train_data.csv"
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
            print(
                f"Warning: _get_state called at step {self.current_step} >= data length {len(self.data)}. Using last valid data."
            )
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
                print(
                    f"Warning: NaN found in column '{col}' at step {self.current_step}. Replacing with 0."
                )
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
        if initial_portfolio_value > 1e-6:
            returns = (
                portfolio_value - initial_portfolio_value
            ) / initial_portfolio_value
        else:
            returns = 0.0

        self.returns_history.append(returns)
        self.portfolio_value = portfolio_value
        self.current_step += 1
        reward = returns
        alpha = 0.05
        min_history_for_cvar = 20
        if len(self.returns_history) >= min_history_for_cvar:
            cvar_penalty_factor = 0.1
            calculated_cvar = self._calculate_cvar(
                self.returns_history[-min_history_for_cvar:], alpha=alpha
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


def train_agent(env, agent, num_episodes=1000, max_steps_per_episode=10000):
    returns_history = []
    losses_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_losses = []
        done = False
        steps_in_episode = 0

        while not done and steps_in_episode < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            loss_val = agent.optimize_model()
            if loss_val is not None:
                episode_losses.append(loss_val)
            steps_in_episode += 1

        returns_history.append(env.portfolio_value)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_history.append(avg_loss)
        print(
            f"Episode {episode + 1}/{num_episodes}, "
            f"Steps: {steps_in_episode}, "
            f"Total Steps: {agent.steps_done}, "
            f"Return: {env.portfolio_value:.2f}, "
            f"Avg Loss: {avg_loss:.4f}"
        )
    return returns_history, losses_history, agent


def evaluate_agent(
    agent: RainbowDQNAgent, eval_data: pd.DataFrame, initial_balance: float
) -> tuple[list, list]:
    print(
        f"Evaluating agent on data from {eval_data.index.min()} to {eval_data.index.max()}"
    )
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
    print(f"Evaluation complete. Final portfolio value: {portfolio_values[-1]:.2f}")
    return portfolio_values, actions_taken


def calculate_sharpe_ratio(
    portfolio_values: list, risk_free_rate: float = 0.0
) -> float:
    portfolio_values_series = pd.Series(portfolio_values)
    daily_returns = portfolio_values_series.pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    excess_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    annualized_sharpe = sharpe_ratio * np.sqrt(252)
    return annualized_sharpe


def calculate_max_drawdown(portfolio_values: list) -> float:
    portfolio_values_series = pd.Series(portfolio_values)
    if portfolio_values_series.empty:
        return 0.0
    cumulative_max = portfolio_values_series.cummax()
    drawdown = (portfolio_values_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown * 100


def calculate_buy_and_hold(
    eval_data: pd.DataFrame, initial_balance: float
) -> tuple[list, float, float]:
    if eval_data.empty:
        return [initial_balance], 0.0, 0.0
    start_price = eval_data["Close"].iloc[0]
    end_price = eval_data["Close"].iloc[-1]
    num_shares = initial_balance // start_price
    cash_left = initial_balance - (num_shares * start_price)
    bnh_portfolio_values = [initial_balance]
    daily_values = (eval_data["Close"] * num_shares) + cash_left
    bnh_portfolio_values.extend(daily_values.tolist())
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


def load_checkpoint(agent, path):
    try:
        checkpoint = torch.load(path, map_location=agent.device)
        agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps_done = checkpoint.get("steps_done", 0)
        print(
            f"Checkpoint successfully loaded from {path}. Resuming at step {agent.steps_done}."
        )
        agent.target_net.eval()
    except FileNotFoundError:
        print(f"Checkpoint file not found at {path}. Starting training from scratch.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting training from scratch.")


def print_charts(returns_history, losses_history, env, save_path):
    print("Generating training performance plot...")
    if not returns_history or not losses_history:
        print("Skipping plotting as no training history was generated.")
        return

    initial_balance = env.initial_balance
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
        print(f"Plot saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)  # Close the figure after saving to free memory


def evaluate(env, agent, eval_data_path):
    initial_balance = env.initial_balance
    try:
        eval_data = pd.read_csv(eval_data_path, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Evaluation data file not found at {eval_data_path}")
        return
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return

    processed_eval_data = calculate_technical_indicators(eval_data.copy())
    processed_eval_data.dropna(inplace=True)

    if processed_eval_data.empty:
        print("Error: Evaluation data is empty after preprocessing.")
        return

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

    bnh_portfolio_values, bnh_total_return, bnh_mdd = calculate_buy_and_hold(
        processed_eval_data,
        initial_balance,  # Calculate BnH on the same processed test data
    )
    bnh_sharpe = calculate_sharpe_ratio(bnh_portfolio_values)
    print(f"  (BnH Annualized Sharpe Ratio: {bnh_sharpe:.2f})")

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


def save_checkpoint(agent, save_path):
    try:
        torch.save(
            {
                "policy_net_state_dict": agent.policy_net.state_dict(),
                "target_net_state_dict": agent.target_net.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "steps_done": agent.steps_done,
                "v_min": agent.v_min,
                "v_max": agent.v_max,
                "num_atoms": agent.num_atoms,
            },
            save_path,
        )
        print(f"Checkpoint saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


if __name__ == "__main__":

    stock_data = collect_stock_data()
    processed_data = calculate_technical_indicators(stock_data.copy())
    processed_data.dropna(inplace=True)

    print("Dataset shape:", processed_data.shape)
    print("  Date range:", processed_data.index.min(), "to", processed_data.index.max())
    # print(processed_data.head()) # Keep print short

    initial_balance = 100000  # Define initial balance
    train_env = TradingEnvironment(processed_data, initial_balance=initial_balance)
    state_size = train_env.state_size
    action_size = 3

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")
    agent = RainbowDQNAgent(state_size, action_size, device=device_name)

    load_checkpoint_flag = False
    checkpoint_load_path = (
        "/home/kartikeya.agrawal_ug25/RL_Final/rainbow_dqn_agent_checkpoint_new.pth"
    )
    if load_checkpoint_flag:
        load_checkpoint(agent, checkpoint_load_path)

    num_train_episodes = 500
    print(f"Starting training for {num_train_episodes} episodes...")
    returns_history, losses_history, agent = train_agent(
        train_env, agent, num_episodes=num_train_episodes
    )

    plot_save_path = f"/home/kartikeya.agrawal_ug25/RL_Final/output/cumreturns_avgloss_{num_train_episodes}.png"
    print_charts(returns_history, losses_history, train_env, plot_save_path)

    checkpoint_save_path = f"/home/kartikeya.agrawal_ug25/RL_Final/output/rainbow_dqn_agent_checkpoint_{num_train_episodes}.pth"
    save_checkpoint(agent, checkpoint_save_path)

    eval_data_path = "/home/kartikeya.agrawal_ug25/RL_Final/eval_data.csv"
    evaluate(train_env, agent, eval_data_path)
