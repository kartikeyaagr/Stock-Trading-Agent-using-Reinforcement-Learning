import numpy as np


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
