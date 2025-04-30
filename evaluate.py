from agent import RainbowDQNAgent
import pandas as pd
from trading_env import TradingEnvironment
import torch
import numpy as np


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
