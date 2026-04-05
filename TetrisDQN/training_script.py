"""
training_script.py – Main Entry Point for Tetris RL Training (Phase 3)
=======================================================================
Orchestrates the full training and evaluation pipeline for both the base
DQN agent and the advanced Dueling DQN agent:
    1. Load configuration from config.yaml.
    2. Print environment API diagnostics.
    3. Run the selected agent's training loop with logging and checkpointing.
    4. Optionally run a trained agent in pure exploitation mode (--eval).

Usage:
    python training_script.py                             # train DQN (default)
    python training_script.py --agent dueling             # train Dueling DQN
    python training_script.py --eval                      # evaluate DQN
    python training_script.py --eval --agent dueling      # evaluate Dueling DQN
    python training_script.py --config my.yaml            # use custom config
    python training_script.py --eval --model path.pt      # evaluate specific ckpt
"""

import argparse
import os
import sys

import numpy as np
import torch

from utils import load_config, get_device, set_seed, MetricsLogger, save_plot
from environment import make_env
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent


# =========================================================================== #
#  Agent factory                                                               #
# =========================================================================== #

def create_agent(agent_type: str, obs_dim: int, action_dim: int,
                 config: dict, device: torch.device):
    """
    Create and return the appropriate agent based on the agent type string.

    Args:
        agent_type (str): Either 'dqn' or 'dueling'.
        obs_dim (int): Observation space dimensionality.
        action_dim (int): Number of discrete actions.
        config (dict): Full configuration dictionary from config.yaml.
        device (torch.device): Torch device for computations.

    Returns:
        DQNAgent or DuelingDQNAgent: The constructed agent.

    Raises:
        ValueError: If agent_type is not recognised.
    """
    if agent_type == "dqn":
        return DQNAgent(obs_dim, action_dim, config["agent"], device)
    elif agent_type == "dueling":
        return DuelingDQNAgent(obs_dim, action_dim, config["dueling_agent"], device)
    else:
        raise ValueError(f"Unknown agent type: '{agent_type}'. Use 'dqn' or 'dueling'.")


def get_results_dir(agent_type: str, config: dict) -> str:
    """
    Return the results directory for the given agent type.

    Args:
        agent_type (str): Either 'dqn' or 'dueling'.
        config (dict): Full configuration dictionary.

    Returns:
        str: Path to the results directory.
    """
    if agent_type == "dqn":
        return config.get("dqn_results_dir", "dqn_results")
    return config.get("dueling_results_dir", "dueling_dqn_results")


def get_agent_label(agent_type: str) -> str:
    """
    Return a human-readable label for the agent type.

    Args:
        agent_type (str): Either 'dqn' or 'dueling'.

    Returns:
        str: Display label string.
    """
    return "Vanilla DQN" if agent_type == "dqn" else "Dueling DQN"


# =========================================================================== #
#  Environment API Confirmation                                                #
# =========================================================================== #

def print_env_info(env, device: torch.device):
    """
    Print environment API diagnostics to stdout.

    Displays the observation space, action space, and selected compute device.
    Then executes one full environment step to confirm that the API functions
    correctly end-to-end.

    Args:
        env: A TetrisEnv instance (Gymnasium-compatible).
        device (torch.device): The compute device selected for training.
    """
    sep = "=" * 62

    print(sep)
    print("  TETRIS RL ENVIRONMENT – API CONFIRMATION")
    print(sep)

    obs_space = env.observation_space
    act_space = env.action_space

    print("\n[Observation Space]")
    print(f"  Type       : {type(obs_space).__name__}")
    print(f"  Shape      : {obs_space.shape}  ({obs_space.shape[0]} features)")
    print(f"  Dtype      : {obs_space.dtype}")
    print(f"  Low        : {obs_space.low.min():.4f}  (all dimensions)")
    print(f"  High       : {obs_space.high.max():.4f}  (all dimensions)")

    print("\n[Action Space]")
    print(f"  Type       : {type(act_space).__name__}  (discrete, finite)")
    print(f"  N          : {act_space.n} actions")

    print("\n[Compute Device]")
    print(f"  Torch device : {device}")
    if device.type == "cuda":
        print(f"  GPU name     : {torch.cuda.get_device_name(device)}")
        print(f"  CUDA version : {torch.version.cuda}")

    print("\n[Environment Step Verification]")
    obs, info = env.reset(seed=42)
    print(f"  env.reset()  -> obs shape={obs.shape}, dtype={obs.dtype}  OK")

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"  env.step({action})   -> obs shape={obs2.shape}")
    print(f"               reward={reward:.4f}  terminated={terminated}  "
          f"truncated={truncated}")
    print(f"               score={info2['score']}  "
          f"lines_cleared={info2['lines_cleared']}  "
          f"steps={info2['steps']}  OK")

    print(f"\n  OK Environment installed, API confirmed, step successful.")
    print(sep + "\n")


# =========================================================================== #
#  Training loop                                                               #
# =========================================================================== #

def train(config: dict, device: torch.device, agent_type: str):
    """
    Execute the full training loop for the selected agent over multiple episodes.

    At each step: the agent selects an action, the environment returns a
    transition, the transition is stored in the replay buffer, and the agent
    performs a gradient update once the buffer is sufficiently populated.

    Metrics (score, lines cleared, survival length, loss, epsilon) are
    logged every episode and saved to JSON.  Model checkpoints are saved
    at regular intervals.

    Args:
        config (dict): Full configuration dictionary loaded from config.yaml.
        device (torch.device): Torch device for network computations.
        agent_type (str): Either 'dqn' or 'dueling'.
    """
    set_seed(int(config.get("seed", 42)))

    env = make_env(config.get("env", {}))
    print_env_info(env, device)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n
    label      = get_agent_label(agent_type)

    agent  = create_agent(agent_type, obs_dim, action_dim, config, device)
    logger = MetricsLogger()

    results_dir   = get_results_dir(agent_type, config)
    num_episodes  = int(config.get("num_episodes", 5000))
    save_interval = int(config.get("save_interval", 500))
    log_interval  = int(config.get("log_interval", 100))

    os.makedirs(results_dir, exist_ok=True)
    print(f"Training {label} for {num_episodes} episodes ...")
    print(f"  Results dir : {results_dir}")
    print(f"  Batch size  : {agent.batch_size}")
    print(f"  Buffer cap  : {agent.config['buffer_capacity']}")
    print(f"  gamma       : {agent.gamma}")
    print(f"  epsilon     : {agent.epsilon:.3f} -> {agent.epsilon_min:.3f} "
          f"over {int(agent.epsilon_decay_steps)} steps\n")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss   = 0.0
        ep_steps       = 0
        done           = False

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.learn()
            agent.decay_epsilon()

            state           = next_state
            episode_reward += reward
            episode_loss   += loss
            ep_steps       += 1

        logger.log(
            episode=episode,
            score=info["score"],
            lines=info["lines_cleared"],
            steps=ep_steps,
            reward=episode_reward,
            loss=episode_loss / max(1, ep_steps),
            epsilon=agent.epsilon,
        )

        if episode % log_interval == 0:
            avg = logger.rolling_avg(log_interval)
            print(
                f"[{label}] Ep {episode:5d}/{num_episodes}  "
                f"score={avg['score']:8.1f}  "
                f"lines={avg['lines']:5.2f}  "
                f"steps={avg['steps']:6.1f}  "
                f"loss={avg['loss']:.4f}  "
                f"eps={agent.epsilon:.4f}"
            )

        if episode % save_interval == 0:
            prefix = "dqn" if agent_type == "dqn" else "dueling_dqn"
            ckpt = os.path.join(results_dir, f"{prefix}_ep{episode}.pt")
            agent.save_model(ckpt)
            print(f"  -> Checkpoint saved: {ckpt}")

    # ---- End of training --------------------------------------------- #
    prefix = "dqn" if agent_type == "dqn" else "dueling_dqn"
    final_path = os.path.join(results_dir, f"{prefix}_final.pt")
    agent.save_model(final_path)
    print(f"\nTraining complete.  Final model saved to: {final_path}")

    save_plot(logger.history, results_dir, title=f"{label} – Training Curves")
    logger.save(os.path.join(results_dir, "training_history.json"))
    env.close()


# =========================================================================== #
#  Evaluation / deployment                                                     #
# =========================================================================== #

def evaluate(
    config:       dict,
    device:       torch.device,
    agent_type:   str,
    model_path:   str,
    num_episodes: int = 50,
):
    """
    Load a trained model and run it in pure exploitation mode.

    No gradient updates or exploration occur.  Epsilon is set to 0.0
    so the agent always selects the greedy action.  Per-episode score,
    lines cleared, and survival length are reported to stdout, along
    with mean ± standard deviation over all evaluation episodes.

    Args:
        config (dict): Configuration dictionary from config.yaml.
        device (torch.device): Torch device for inference.
        agent_type (str): Either 'dqn' or 'dueling'.
        model_path (str): Path to a checkpoint file.
        num_episodes (int): Number of evaluation episodes to run (default 50).
    """
    render_mode = config.get("render_mode", None)
    env = make_env(config.get("env", {}), render_mode=render_mode)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n
    label      = get_agent_label(agent_type)

    agent = create_agent(agent_type, obs_dim, action_dim, config, device)
    agent.load_model(model_path)
    agent.epsilon = 0.0  # pure greedy

    print(f"Evaluating {label} model: {model_path}")
    print(f"Episodes: {num_episodes}  |  render_mode: {render_mode}\n")

    scores, lines_list, steps_list, rewards_list = [], [], [], []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        scores.append(info["score"])
        lines_list.append(info["lines_cleared"])
        steps_list.append(info["steps"])
        rewards_list.append(ep_reward)

        print(
            f"  Ep {ep:3d}: score={info['score']:6d}  "
            f"lines={info['lines_cleared']:4d}  "
            f"steps={info['steps']:6d}  "
            f"reward={ep_reward:.1f}"
        )

    print(f"\n{'='*50}")
    print(f"  {label} – Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Mean score  : {np.mean(scores):8.1f} ± {np.std(scores):.1f}")
    print(f"  Mean lines  : {np.mean(lines_list):8.2f} ± {np.std(lines_list):.2f}")
    print(f"  Mean steps  : {np.mean(steps_list):8.1f} ± {np.std(steps_list):.1f}")
    print(f"  Mean reward : {np.mean(rewards_list):8.1f} ± {np.std(rewards_list):.1f}")
    env.close()


# =========================================================================== #
#  Entry point                                                                 #
# =========================================================================== #

def main():
    """
    Parse command-line arguments and dispatch to training or evaluation mode.

    Flags:
        --config  PATH       Path to config.yaml  (default: 'config.yaml').
        --agent   TYPE       'dqn' (default) or 'dueling'.
        --eval               Run evaluation instead of training.
        --model   PATH       Checkpoint for evaluation (auto-detected if omitted).
        --eval-episodes N    Number of evaluation episodes (default from config).
    """
    parser = argparse.ArgumentParser(description="Tetris RL Agent – Phase 3")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--agent", type=str, default="dqn", choices=["dqn", "dueling"],
        help="Agent type: 'dqn' (vanilla DQN) or 'dueling' (Dueling DQN)."
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run in evaluation mode (load saved model, no training)."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a model checkpoint (used with --eval)."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=None,
        help="Number of episodes to run in evaluation mode."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))

    if args.eval:
        # Auto-detect model path if not specified
        if args.model is None:
            results_dir = get_results_dir(args.agent, config)
            prefix = "dqn" if args.agent == "dqn" else "dueling_dqn"
            args.model = os.path.join(results_dir, f"{prefix}_final.pt")

        num_eval = args.eval_episodes or int(config.get("eval_episodes", 50))
        evaluate(config, device, args.agent, args.model, num_eval)
    else:
        train(config, device, args.agent)


if __name__ == "__main__":
    main()
