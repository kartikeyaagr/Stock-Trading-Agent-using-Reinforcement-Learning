import matplotlib as plt


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
