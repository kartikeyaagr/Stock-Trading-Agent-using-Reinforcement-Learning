# Rainbow DQN Stock Trading Agent with CVaR Optimization

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)

---

## 1. Project Overview

This repository contains a sophisticated Deep Reinforcement Learning (DRL) agent designed for algorithmic stock trading. The agent is built on the **Rainbow DQN** algorithm, integrating seven state-of-the-art improvements to learn a robust trading policy for a single stock.

A key feature of this implementation is its modularity and extensive integration with **Weights & Biases (W&B)** for comprehensive experiment tracking, hyperparameter tuning, and results visualization. Furthermore, the agent's reward function can be augmented with a **Conditional Value at Risk (CVaR)** penalty, providing a mechanism to train for risk-adjusted returns rather than myopic profit maximization.

This project was developed for academic research and serves as a strong foundation for further exploration into DRL applications in finance. For a detailed explanation of the methodology, theory, and experimental results, please refer to the academic paper in the repository.

## 2. Core Features

- **Advanced DRL Agent:** A complete implementation of the Rainbow DQN algorithm.
  - Distributional Reinforcement Learning (C51)
  - Dueling Network Architecture
  - Prioritized Experience Replay (PER)
  - N-Step Bootstrapping
  - Noisy Networks for Exploration
  - Double Q-Learning
- **Comprehensive Experiment Tracking:** Deeply integrated with [Weights & Biases](https://wandb.ai) for logging metrics, hyperparameters, system usage, and evaluation plots.
- **Flexible Hyperparameter Management:** Centralized `config` dictionary for easy tuning of all major parameters.
- **Risk-Aware Learning:** Optional CVaR penalty in the reward function to promote risk-averse strategies.
- **Robust Evaluation Suite:** Generates a suite of analytical plots and metrics for out-of-sample testing, including portfolio value, drawdown, trading actions, and returns distribution.

## 3. Design Philosophy and Model Choices

The architecture of this project was deliberately chosen to address the specific challenges of applying reinforcement learning to financial markets.

**Why Reinforcement Learning?**
Algorithmic trading is fundamentally a sequential decision-making problem. An agent must make a series of `Buy`, `Sell`, or `Hold` decisions over time to maximize a long-term objective (e.g., cumulative return). RL provides a natural framework for learning such policies through trial-and-error interaction with a simulated market environment.

**Why Rainbow DQN?**
While a simple DQN could be used, financial markets are notoriously noisy and non-stationary. A more powerful and stable algorithm is required to extract a meaningful signal. Rainbow DQN was chosen because it combines several orthogonal improvements that each address a specific weakness of the original DQN algorithm:

- **Dueling DQN:** Financial state representations can have high value independent of the action taken (e.g., in a strong bull market). The Dueling architecture allows the model to learn the state's value (`V(s)`) separately from the advantage of each action (`A(s,a)`), leading to better policy evaluation.
- **Prioritized Experience Replay (PER):** Market data contains long periods of low-signal movement punctuated by critical, high-signal events (e.g., crashes, breakouts). PER allows the agent to focus its learning on these more "surprising" and informative transitions, leading to greater sample efficiency.
- **Noisy Nets:** Effective exploration is critical. Instead of random `ε-greedy` actions, Noisy Nets introduce learnable noise directly into the network's weights. This allows the agent to learn a more sophisticated, state-dependent exploration strategy, which is more suitable than random exploration in a structured environment like finance.
- **Distributional RL (C51):** This is perhaps the most crucial component for finance. Instead of learning a single average expected return (Q-value), the agent learns the full probability distribution of potential returns. This allows it to understand not just the expected outcome but also the risk (i.e., the variance, skew, and tail properties of the return distribution) associated with an action.
- **N-Step Returns & Double Q-Learning:** These components provide faster propagation of reward information and increased learning stability, respectively, which are beneficial in any complex RL problem.

**Why CVaR-Adjusted Rewards?**
Maximizing raw returns often leads to high-risk strategies with severe drawdowns. In real-world finance, managing risk is paramount. Conditional Value at Risk (CVaR) measures the expected loss in the worst-case scenarios. By adding a penalty based on CVaR to the reward signal, we explicitly incentivize the agent to learn policies that avoid strategies with a high potential for large losses, aiming for a better risk-adjusted performance (e.g., a higher Sharpe Ratio).

## 4. Project Structure

The repository is organized into a main script containing all components, alongside data and output directories.

```
.
├── RL.py                 # Main script with all classes (Environment, Agent, etc.) and logic
├── data/
│   ├── train_data.csv    # Training dataset (must be created by user)
│   └── eval_data.csv     # Evaluation dataset (must be created by user)
├── output/
│   ├── (checkpoints)     # Saved model checkpoints (.pth) will appear here
│   └── (plots)           # Saved performance plots (.png) will appear here
├── job_script.sh         # Example Slurm job submission script for HPC execution
└── requirements.txt      # Python package dependencies
```

## 5. How to Run

### Configure the Experiment
Open `RL.py` and modify the `config` dictionary inside the `if __name__ == "__main__":` block at the bottom of the file. This is where you control all aspects of the run:
- Set paths for training/evaluation data and output directories.
- Define training parameters like `num_train_episodes` and `learning_rate`.
- Adjust agent architecture parameters like `hidden_size` or C51 settings (`v_min`, `v_max`).
- Toggle the CVaR penalty by changing `cvar_penalty_factor`.

### Execute the Training
- **Locally or on a login node:**
  ```bash
  python RL.py
  ```
- **On an HPC using a scheduler (e.g., Slurm):**
  - Customize the `job_script.sh` file with your specific HPC resource requests (GPU, CPUs, memory, walltime), module loads, and conda environment name.
  - Submit the job:
    ```bash
    sbatch job_script.sh
    ```

### Monitor and Analyze Results
1.  **Console Output:** The script will print episode-by-episode progress to the console or your job's log file.
2.  **Weights & Biases Dashboard:** The most powerful tool. After starting a run, a W&B URL will be printed. Open this link in a browser to see live charts of portfolio value, loss, Q-values, and more.
3.  **Local Files:** The `output/` directory will contain the final saved model checkpoint and all generated plots.

## 6. Limitations

- **Simulation Only:** Tested purely in simulation based on historical data. Does not account for real-world factors like transaction costs, slippage, latency, or market impact of trades.
- **Single Stock:** Designed and evaluated for only one specific stock. Performance may vary significantly on other assets or market conditions.
- **Hyperparameter Sensitivity:** Performance is dependent on the chosen hyperparameters, which may require tuning for different datasets or objectives.
- **Data Snooping:** While a train/test split was used, the selection of technical indicators and model architecture could implicitly involve some lookahead bias.
