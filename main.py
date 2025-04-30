# LIBRARY IMPORTS
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# LOCAL IMPORTS
from data_wrangling import collect_stock_data, calculate_technical_indicators
from trading_env import TradingEnvironment
from noisylinear import NoisyLinear
from rdqn import RainbowDQN
from replaybuffer import PrioritizedReplayBuffer
from agent import RainbowDQNAgent
from train import train_agent
from evaluate import (
    evaluate_agent,
    calculate_buy_and_hold,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)

if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    stock_data = collect_stock_data()
    processed_data = calculate_technical_indicators(stock_data.copy())
    processed_data.dropna(inplace=True)

    print("Dataset shape:", processed_data.shape)
    print("  Date range:", processed_data.index.min(), "to", processed_data.index.max())

    env = TradingEnvironment(processed_data, initial_balance=100000)
    state_size = env.state_size
    action_size = 3

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")

    agent = RainbowDQNAgent(state_size, action_size, device=device_name)

    load_checkpoint = False
    load_path = (
        "/home/kartikeya.agrawal_ug25/RL_Final/rainbow_dqn_agent_checkpoint_new.pth"
    )

    if load_checkpoint:
        try:
            # Map location ensures model loads correctly even if saved on GPU and loading on CPU
            checkpoint = torch.load(load_path, map_location=agent.device)
            agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            agent.steps_done = checkpoint.get("steps_done", 0)
            print(
                f"Checkpoint successfully loaded from {load_path}. Resuming at step {agent.steps_done}."
            )

            agent.target_net.eval()
        except FileNotFoundError:
            print(
                f"Checkpoint file not found at {load_path}. Starting training from scratch."
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")

    num_train_episodes = 500
    print(f"Starting training for {num_train_episodes} episodes...")
    returns_history, losses_history, agent = train_agent(
        env, agent, num_episodes=num_train_episodes
    )

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

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Average Loss", color=color)
    ax2.plot(losses_history, color=color, alpha=0.6, label="Avg Loss")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.title("Training Performance: Portfolio Value and Average Loss per Episode")
    plt.savefig(
        "/home/kartikeya.agrawal_ug25/RL_Final/plot.png", dpi=300, transparent=True
    )

    # --- Checkpoint Saving ---
    save_path = "/home/kartikeya.agrawal_ug25/RL_Final/rainbow_dqn_agent_checkpoint_final.pth"  # Example path
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
