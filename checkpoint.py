import torch


def load_checkpoint(agent, path):
    try:
        checkpoint = torch.load(path, map_location=agent.device)
        agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps_done = checkpoint.get("steps_done", 0)
        print(
            f"Checkpoint successfully loaded from {path}. Resuming at step {agent.steps_done}."
        )
        agent.target_net.eval()
    except FileNotFoundError:
        print(f"Checkpoint file not found at {path}. Starting training from scratch.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting training from scratch.")


def save_checkpoint(agent, save_path):
    try:
        torch.save(
            {
                "policy_net_state_dict": agent.policy_net.state_dict(),
                "target_net_state_dict": agent.target_net.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "steps_done": agent.steps_done,
                "v_min": agent.v_min,
                "v_max": agent.v_max,
                "num_atoms": agent.num_atoms,
            },
            save_path,
        )
        print(f"Checkpoint saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
