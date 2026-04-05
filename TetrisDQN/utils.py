"""
utils.py – Shared Utilities for the DQN Tetris Project
=======================================================
Provides helpers used by both training_script.py and any future algorithm
modules.  Keeping these functions here avoids code duplication and makes
each algorithm file (dqn_agent.py, dueling_dqn_agent.py, etc.) lean.

Responsibilities:
    • Configuration loading from YAML
    • Device selection (CPU / CUDA / MPS)
    • Random seed management
    • Per-episode metrics logging with rolling averages
    • Training curve plotting and history persistence
"""

import json
import os
import random

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #

def load_config(path: str = "config.yaml") -> dict:
    """
    Load a YAML configuration file and return its contents as a dict.

    All hyperparameters, paths, and runtime flags are centralised in
    config.yaml to avoid hard-coded values scattered across source files.

    Args:
        path (str): Path to the YAML file. Defaults to 'config.yaml'.

    Returns:
        dict: Parsed configuration mapping.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: '{path}'")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
#  Device selection                                                            #
# --------------------------------------------------------------------------- #

def get_device(preference: str = "auto") -> torch.device:
    """
    Resolve and return the best available compute device.

    Priority order for 'auto': CUDA > MPS (Apple Silicon) > CPU.

    Args:
        preference (str): One of 'auto', 'cpu', 'cuda', or 'mps'.

    Returns:
        torch.device: The selected device.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# --------------------------------------------------------------------------- #
#  Reproducibility                                                             #
# --------------------------------------------------------------------------- #

def set_seed(seed: int):
    """
    Set random seeds for Python, NumPy, and PyTorch to enable reproducibility.

    Args:
        seed (int): Integer seed value.  Use the same value across runs to
            reproduce results exactly (subject to hardware non-determinism).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Metrics logging                                                             #
# --------------------------------------------------------------------------- #

class MetricsLogger:
    """
    Accumulates per-episode training metrics and computes rolling averages.

    Tracks: episode index, in-game score, lines cleared, survival length
    (steps), cumulative episode reward, mean training loss, and epsilon.
    Results can be saved to JSON and plotted via save_plot().
    """

    def __init__(self):
        """
        Initialise an empty history store for all tracked metrics.
        """
        self.history: dict = {
            "episode": [],
            "score":   [],
            "lines":   [],
            "steps":   [],
            "reward":  [],
            "loss":    [],
            "epsilon": [],
        }

    def log(
        self,
        episode: int,
        score:   int,
        lines:   int,
        steps:   int,
        reward:  float,
        loss:    float,
        epsilon: float,
    ):
        """
        Append one episode's worth of metrics to the history store.

        Args:
            episode (int): Episode index (1-based).
            score (int): Total in-game Tetris score for the episode.
            lines (int): Total number of lines cleared during the episode.
            steps (int): Number of environment steps taken (survival length).
            reward (float): Cumulative undiscounted reward for the episode.
            loss (float): Mean gradient-update loss for the episode.
            epsilon (float): Exploration rate at the end of the episode.
        """
        self.history["episode"].append(episode)
        self.history["score"].append(score)
        self.history["lines"].append(lines)
        self.history["steps"].append(steps)
        self.history["reward"].append(reward)
        self.history["loss"].append(loss)
        self.history["epsilon"].append(epsilon)

    def rolling_avg(self, window: int = 100) -> dict:
        """
        Compute the mean of each metric over the last `window` episodes.

        Args:
            window (int): Number of recent episodes to include in the average.

        Returns:
            dict: Same keys as history; values are scalar float averages.
        """
        result = {}
        for key, vals in self.history.items():
            arr = np.array(vals[-window:], dtype=float)
            result[key] = float(np.mean(arr)) if len(arr) > 0 else 0.0
        return result

    def save(self, path: str):
        """
        Write the full training history to a JSON file.

        Args:
            path (str): Output file path (e.g. 'dqn_results/history.json').
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {k: [float(x) for x in v] for k, v in self.history.items()},
                f,
                indent=2,
            )


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #

def save_plot(history: dict, results_dir: str, title: str = "DQN Tetris – Training Curves"):
    """
    Generate and save training curve plots to the results directory.

    Creates a three-panel figure:
        1. Episode score (primary metric, Phase 1 §Evaluation Metric)
        2. Lines cleared per episode (secondary metric)
        3. Survival steps per episode (tertiary metric)

    Each panel shows the raw per-episode value (faint) and a 100-episode
    rolling average (solid), matching the evaluation protocol in Phase 1.

    Also persists the raw history dict to a JSON file for later analysis.

    Args:
        history (dict): Training history from MetricsLogger.history.
        results_dir (str): Directory where the PNG and JSON are saved.
        title (str): Title for the figure (default: 'DQN Tetris – Training Curves').
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [utils] matplotlib not installed — skipping plot generation.")
        return

    os.makedirs(results_dir, exist_ok=True)

    episodes = np.array(history["episode"], dtype=float)
    window = 100

    def rolling_avg(arr: np.ndarray, w: int) -> np.ndarray:
        """Compute a causal rolling mean; first w-1 values are NaN."""
        result = np.convolve(arr, np.ones(w) / w, mode="valid")
        return np.concatenate([np.full(w - 1, np.nan), result])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    plot_cfg = [
        ("score",  "Episode Score (in-game)",    "steelblue"),
        ("lines",  "Lines Cleared per Episode",  "seagreen"),
        ("steps",  "Survival Steps per Episode", "darkorange"),
    ]

    for ax, (key, label, color) in zip(axes, plot_cfg):
        raw    = np.array(history[key], dtype=float)
        rolled = rolling_avg(raw, window)
        ax.plot(episodes, raw,    alpha=0.2, color=color, linewidth=0.5)
        ax.plot(episodes, rolled, color=color, linewidth=1.5,
                label=f"{label} ({window}-ep avg)")
        ax.set_ylabel(label)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  -> Plot saved:    {plot_path}")

    # Persist raw history
    hist_path = os.path.join(results_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(
            {k: [float(x) for x in v] for k, v in history.items()},
            f,
            indent=2,
        )
    print(f"  -> History saved: {hist_path}")
