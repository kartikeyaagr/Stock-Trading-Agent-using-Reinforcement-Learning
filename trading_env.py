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
