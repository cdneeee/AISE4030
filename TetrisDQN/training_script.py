"""
training_script.py – Main Entry Point for DQN Tetris Training
==============================================================
Orchestrates the full training pipeline:
    1. Load configuration from config.yaml.
    2. Print environment API diagnostics (Task 1 – Phase 2 requirement).
    3. Run the DQN training loop with logging and checkpointing.
    4. Optionally run the trained agent in pure exploitation mode (--eval).

Usage:
    python training_script.py                        # train from scratch
    python training_script.py --eval                 # evaluate saved model
    python training_script.py --config my.yaml       # use custom config
    python training_script.py --eval --model path.pt # evaluate specific ckpt
"""

import argparse
import os
import sys

import numpy as np
import torch

from utils import load_config, get_device, set_seed, MetricsLogger, save_plot
from environment import make_env
from dqn_agent import DQNAgent


# =========================================================================== #
#  Task 1 – Environment API Confirmation                                       #
# =========================================================================== #

def print_env_info(env, device: torch.device):
    """
    Print environment API diagnostics to stdout (Phase 2, Task 1).

    Displays the observation space, action space, and selected compute device.
    Then executes one full environment step to confirm that the API functions
    correctly end-to-end.  The console output of this function must be captured
    and included in the Phase 2 progress report.

    Args:
        env: A TetrisEnv instance (Gymnasium-compatible).
        device (torch.device): The compute device selected for training.
    """
    sep = "=" * 62

    print(sep)
    print("  TETRIS RL ENVIRONMENT – API CONFIRMATION (Phase 2, Task 1)")
    print(sep)

    obs_space = env.observation_space
    act_space = env.action_space

    # ---- Observation space ------------------------------------------- #
    print("\n[Observation Space]")
    print(f"  Type       : {type(obs_space).__name__}")
    print(f"  Shape      : {obs_space.shape}  ({obs_space.shape[0]} features)")
    print(f"  Dtype      : {obs_space.dtype}")
    print(f"  Low        : {obs_space.low.min():.4f}  (all dimensions)")
    print(f"  High       : {obs_space.high.max():.4f}  (all dimensions)")
    print(f"  Breakdown  :")
    print(f"    Board occupancy (20×10, binary) : {env.ROWS * env.COLS:>3} values")
    print(f"    Current piece  (one-hot, 7)     :   7 values")
    print(f"    Orientation    (one-hot, 4)     :   4 values")
    print(f"    Column position (normalised)    :   1 value")
    print(f"    Row position    (normalised)    :   1 value")
    print(f"    Next piece     (one-hot, 7)     :   7 values")
    print(f"    ─────────────────────────────────────────")
    print(f"    Total                           : {obs_space.shape[0]:>3} float32 ∈ [0, 1]")

    # ---- Action space ------------------------------------------------- #
    print("\n[Action Space]")
    print(f"  Type       : {type(act_space).__name__}  (discrete, finite)")
    print(f"  N          : {act_space.n} actions")
    print(f"  Actions    :")
    actions = {
        0: "LEFT    – move piece one column left",
        1: "RIGHT   – move piece one column right",
        2: "ROT_CW  – rotate 90° clockwise",
        3: "ROT_CCW – rotate 90° counter-clockwise",
        4: "DROP    – hard-drop to lowest valid row",
        5: "NO_OP   – no action (gravity only)",
    }
    for idx, desc in actions.items():
        print(f"    {idx}: {desc}")

    # ---- Device ------------------------------------------------------- #
    print("\n[Compute Device]")
    print(f"  Torch device : {device}")
    if device.type == "cuda":
        print(f"  GPU name     : {torch.cuda.get_device_name(device)}")
        print(f"  CUDA version : {torch.version.cuda}")
    elif device.type == "mps":
        print(f"  Backend      : Apple MPS (Metal Performance Shaders)")
    else:
        print(f"  Backend      : CPU (no GPU acceleration)")

    # ---- Step verification -------------------------------------------- #
    print("\n[Environment Step Verification]")
    obs, info = env.reset(seed=42)
    print(f"  env.reset()  → obs shape={obs.shape}, dtype={obs.dtype}  ✔")

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"  env.step({action})   → obs shape={obs2.shape}")
    print(f"               reward={reward:.4f}  terminated={terminated}  "
          f"truncated={truncated}")
    print(f"               score={info2['score']}  "
          f"lines_cleared={info2['lines_cleared']}  "
          f"steps={info2['steps']}  ✔")

    print(f"\n  ✔ Environment installed, API confirmed, step successful.")
    print(sep + "\n")


# =========================================================================== #
#  Training loop                                                               #
# =========================================================================== #

def train(config: dict, device: torch.device):
    """
    Execute the full DQN training loop over multiple episodes.

    At each step: the agent selects an action, the environment returns a
    transition, the transition is stored in the replay buffer, and the agent
    performs a gradient update once the buffer is sufficiently populated.

    Metrics (score, lines cleared, survival length, loss, epsilon) are
    logged every episode.  A rolling 100-episode average is printed every
    log_interval episodes.  Model checkpoints are saved every save_interval
    episodes and a final checkpoint is saved at the end.

    Args:
        config (dict): Full configuration dictionary loaded from config.yaml.
        device (torch.device): Torch device for network computations.
    """
    set_seed(int(config.get("seed", 42)))

    env = make_env(config.get("env", {}))
    print_env_info(env, device)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent  = DQNAgent(obs_dim, action_dim, config["agent"], device)
    logger = MetricsLogger()

    results_dir   = config.get("results_dir", "dqn_results")
    num_episodes  = int(config.get("num_episodes", 5000))
    save_interval = int(config.get("save_interval", 500))
    log_interval  = int(config.get("log_interval", 100))

    os.makedirs(results_dir, exist_ok=True)
    print(f"Training DQN for {num_episodes} episodes …")
    print(f"  Results dir : {results_dir}")
    print(f"  Batch size  : {agent.batch_size}")
    print(f"  Buffer cap  : {config['agent']['buffer_capacity']}")
    print(f"  gamma       : {agent.gamma}")
    print(f"  epsilon     : {agent.epsilon:.3f} → {agent.epsilon_min:.3f} "
          f"over {int(agent.epsilon_decay_steps)} steps\n")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss   = 0.0
        ep_steps       = 0
        done           = False

        while not done:
            action                          = agent.choose_action(state, training=True)
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
                f"Ep {episode:5d}/{num_episodes}  "
                f"score={avg['score']:8.1f}  "
                f"lines={avg['lines']:5.2f}  "
                f"steps={avg['steps']:6.1f}  "
                f"loss={avg['loss']:.4f}  "
                f"eps={agent.epsilon:.4f}"
            )

        if episode % save_interval == 0:
            ckpt = os.path.join(results_dir, f"dqn_ep{episode}.pt")
            agent.save_model(ckpt)
            print(f"  → Checkpoint saved: {ckpt}")

    # ---- End of training --------------------------------------------- #
    final_path = os.path.join(results_dir, "dqn_final.pt")
    agent.save_model(final_path)
    print(f"\nTraining complete.  Final model saved to: {final_path}")

    save_plot(logger.history, results_dir)
    logger.save(os.path.join(results_dir, "training_history.json"))
    env.close()


# =========================================================================== #
#  Evaluation / deployment                                                     #
# =========================================================================== #

def evaluate(
    config:       dict,
    device:       torch.device,
    model_path:   str,
    num_episodes: int = 10,
):
    """
    Load a trained DQN model and run it in pure exploitation mode.

    No gradient updates or exploration occur.  Epsilon is set to 0.0
    so the agent always selects the greedy action.  Per-episode score,
    lines cleared, and survival length are reported to stdout.

    Args:
        config (dict): Configuration dictionary from config.yaml.
        device (torch.device): Torch device for inference.
        model_path (str): Path to a checkpoint file saved by DQNAgent.save_model().
        num_episodes (int): Number of evaluation episodes to run (default 10).
    """
    render_mode = config.get("render_mode", None)
    env = make_env(config.get("env", {}), render_mode=render_mode)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim, config["agent"], device)
    agent.load_model(model_path)
    agent.epsilon = 0.0  # pure greedy

    print(f"Evaluating model: {model_path}")
    print(f"Episodes: {num_episodes}  |  render_mode: {render_mode}\n")

    scores, lines_list, steps_list = [], [], []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, training=False)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        scores.append(info["score"])
        lines_list.append(info["lines_cleared"])
        steps_list.append(info["steps"])

        print(
            f"  Ep {ep:3d}: score={info['score']:6d}  "
            f"lines={info['lines_cleared']:4d}  "
            f"steps={info['steps']:6d}"
        )

    print(f"\nSummary over {num_episodes} episodes:")
    print(f"  Mean score  : {np.mean(scores):.1f}")
    print(f"  Mean lines  : {np.mean(lines_list):.2f}")
    print(f"  Mean steps  : {np.mean(steps_list):.1f}")
    env.close()


# =========================================================================== #
#  Entry point                                                                 #
# =========================================================================== #

def main():
    """
    Parse command-line arguments and dispatch to the training or evaluation mode.

    Flags:
        --config  PATH   Path to config.yaml  (default: 'config.yaml').
        --eval           Run evaluation instead of training.
        --model   PATH   Checkpoint for evaluation (default: dqn_results/dqn_final.pt).
        --eval-episodes  Number of evaluation episodes (default: 10).
    """
    parser = argparse.ArgumentParser(description="DQN Tetris Agent – Phase 2")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run in evaluation mode (load saved model, no training)."
    )
    parser.add_argument(
        "--model", type=str, default="dqn_results/dqn_final.pt",
        help="Path to a model checkpoint (used with --eval)."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of episodes to run in evaluation mode."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))

    if args.eval:
        evaluate(config, device, args.model, args.eval_episodes)
    else:
        train(config, device)


if __name__ == "__main__":
    main()
