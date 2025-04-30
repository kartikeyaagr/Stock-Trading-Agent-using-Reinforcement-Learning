# --- EXTERNAL IMPORTS ---
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# --- INTERNAL MODULES ---
from data_processing import collect_stock_data, calculate_technical_indicators
from trading_env import TradingEnvironment
from noise import NoisyLinear
from dqn import PrioritizedReplayBuffer, RainbowDQN
from agent import RainbowDQNAgent
from train import train_agent
from eval import (
    evaluate,
    evaluate_agent,
    calculate_buy_and_hold,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from checkpoint import save_checkpoint, load_checkpoint
from charts import print_charts

# ! We can choose to set random seeds for reproducibility
"""np.random.seed(42)
torch.manual_seed(42)
random.seed(42)"""

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
checkpoint_load_path = "checkpoints/rainbow_dqn_agent_checkpoint_new.pth"
if load_checkpoint_flag:
    load_checkpoint(agent, checkpoint_load_path)

num_train_episodes = 500
print(f"Starting training for {num_train_episodes} episodes...")
returns_history, losses_history, agent = train_agent(
    train_env, agent, num_episodes=num_train_episodes
)

plot_save_path = f"output/cumreturns_avgloss_{num_train_episodes}.png"
print_charts(returns_history, losses_history, train_env, plot_save_path)

checkpoint_save_path = (
    f"checkpoints/rainbow_dqn_agent_checkpoint_{num_train_episodes}.pth"
)
save_checkpoint(agent, checkpoint_save_path)

eval_data_path = "data/eval_data.csv"
evaluate(train_env, agent, eval_data_path)
