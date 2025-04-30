import pandas as pd
import numpy as np


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
