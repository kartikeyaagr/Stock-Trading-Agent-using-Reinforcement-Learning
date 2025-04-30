from dqn import RainbowDQN
import pandas as pd
import numpy as np
from agent import RainbowDQNAgent
from trading_env import TradingEnvironment
import torch
from data_processing import calculate_technical_indicators


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
