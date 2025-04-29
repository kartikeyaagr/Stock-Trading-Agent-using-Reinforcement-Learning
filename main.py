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
