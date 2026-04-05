"""
compare.py – Comparative Analysis: DQN vs Dueling DQN (Phase 3, Task 3)
=========================================================================
Loads training histories from both agents and generates publication-quality
comparison plots for the four required metrics:

    1. Learning Speed    – Overlaid reward curves with threshold marker
    2. Loss Convergence  – Overlaid loss curves with smoothing
    3. Final Performance – Bar chart of mean ± std over evaluation episodes
    4. Stability         – Reward curves with shaded variance regions

All plots use a consistent colour scheme:
    - DQN (base):    blue (#4C72B0)
    - Dueling DQN:   orange (#DD8452)

Usage:
    python compare.py                                     # default paths
    python compare.py --dqn-hist dqn_results/training_history.json \
                      --dueling-hist dueling_dqn_results/training_history.json
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file saving
import matplotlib.pyplot as plt


# ---- Consistent style ------------------------------------------------------- #
COLOR_DQN     = "#4C72B0"
COLOR_DUELING = "#DD8452"
FONT_SIZE     = 12
SMOOTH_WINDOW = 100  # rolling average window

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "figure.dpi": 150,
})


# ----------------------------------------------------------------------------- #
#  Helpers                                                                       #
# ----------------------------------------------------------------------------- #

def load_history(path: str) -> dict:
    """
    Load a training history JSON file.

    Args:
        path (str): Path to the JSON file produced by MetricsLogger.save().

    Returns:
        dict: Training history with keys: episode, score, lines, steps,
              reward, loss, epsilon.
    """
    with open(path, "r") as f:
        return json.load(f)


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a causal rolling mean; first (window-1) values are NaN.

    If the array is shorter than the window, uses the array length instead.

    Args:
        arr (np.ndarray): 1-D array of values.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Smoothed array of the same length as arr.
    """
    w = min(window, len(arr))
    result = np.convolve(arr, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), result])


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a causal rolling standard deviation; first (window-1) values are NaN.

    If the array is shorter than the window, uses the array length instead.

    Args:
        arr (np.ndarray): 1-D array of values.
        window (int): Rolling window size.

    Returns:
        np.ndarray: Rolling std array of the same length as arr.
    """
    w = min(window, len(arr))
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.std(arr[i - w + 1: i + 1])
    return out


# ----------------------------------------------------------------------------- #
#  Plot 1: Learning Speed (overlaid reward curves)                              #
# ----------------------------------------------------------------------------- #

def plot_learning_speed(dqn_hist: dict, dueling_hist: dict, output_dir: str):
    """
    Overlay smoothed episode reward curves for both agents.

    Marks the episode at which each agent first crosses a reward threshold
    (defined as the max of the two agents' 75th-percentile rolling reward).

    Args:
        dqn_hist (dict): DQN training history.
        dueling_hist (dict): Dueling DQN training history.
        output_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for hist, color, label in [
        (dqn_hist, COLOR_DQN, "Vanilla DQN"),
        (dueling_hist, COLOR_DUELING, "Dueling DQN"),
    ]:
        eps = np.array(hist["episode"])
        raw = np.array(hist["reward"])
        smoothed = rolling_mean(raw, SMOOTH_WINDOW)
        ax.plot(eps, raw, alpha=0.12, color=color, linewidth=0.5)
        ax.plot(eps, smoothed, color=color, linewidth=2, label=label)

    # Threshold line: median of the better agent's final 100 episodes
    dqn_final = np.mean(dqn_hist["reward"][-100:])
    dueling_final = np.mean(dueling_hist["reward"][-100:])
    threshold = max(dqn_final, dueling_final) * 0.5
    ax.axhline(threshold, color="gray", linestyle="--", alpha=0.6,
               label=f"50% of best final ({threshold:.0f})")

    # Mark first crossing
    for hist, color, label in [
        (dqn_hist, COLOR_DQN, "DQN"),
        (dueling_hist, COLOR_DUELING, "Dueling"),
    ]:
        smoothed = rolling_mean(np.array(hist["reward"]), SMOOTH_WINDOW)
        crossed = np.where(smoothed >= threshold)[0]
        if len(crossed) > 0:
            ep_cross = hist["episode"][crossed[0]]
            ax.axvline(ep_cross, color=color, linestyle=":", alpha=0.6)
            ax.annotate(f"{label}: ep {ep_cross}", xy=(ep_cross, threshold),
                        fontsize=9, color=color,
                        xytext=(10, 10), textcoords="offset points")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Episode Reward")
    ax.set_title("Learning Speed Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_learning_speed.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> Saved: {path}")


# ----------------------------------------------------------------------------- #
#  Plot 2: Loss Convergence                                                     #
# ----------------------------------------------------------------------------- #

def plot_loss_convergence(dqn_hist: dict, dueling_hist: dict, output_dir: str):
    """
    Overlay smoothed Q-network loss curves for both agents.

    Args:
        dqn_hist (dict): DQN training history.
        dueling_hist (dict): Dueling DQN training history.
        output_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for hist, color, label in [
        (dqn_hist, COLOR_DQN, "Vanilla DQN"),
        (dueling_hist, COLOR_DUELING, "Dueling DQN"),
    ]:
        eps = np.array(hist["episode"])
        raw = np.array(hist["loss"])
        smoothed = rolling_mean(raw, SMOOTH_WINDOW)
        ax.plot(eps, raw, alpha=0.12, color=color, linewidth=0.5)
        ax.plot(eps, smoothed, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean MSE Loss (per episode)")
    ax.set_title("Loss Convergence Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_loss_convergence.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> Saved: {path}")


# ----------------------------------------------------------------------------- #
#  Plot 3: Final Performance                                                    #
# ----------------------------------------------------------------------------- #

def plot_final_performance(dqn_hist: dict, dueling_hist: dict, output_dir: str,
                           eval_window: int = 100):
    """
    Bar chart comparing mean ± std of final evaluation-equivalent performance.

    Uses the last `eval_window` training episodes as a proxy if separate
    evaluation logs are unavailable.

    Args:
        dqn_hist (dict): DQN training history.
        dueling_hist (dict): Dueling DQN training history.
        output_dir (str): Directory to save the plot.
        eval_window (int): Number of final episodes to use (default 100).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics = [
        ("reward", "Cumulative Reward"),
        ("score", "In-Game Score"),
        ("lines", "Lines Cleared"),
    ]

    for ax, (key, ylabel) in zip(axes, metrics):
        dqn_vals = np.array(dqn_hist[key][-eval_window:])
        duel_vals = np.array(dueling_hist[key][-eval_window:])

        means = [np.mean(dqn_vals), np.mean(duel_vals)]
        stds  = [np.std(dqn_vals), np.std(duel_vals)]

        bars = ax.bar(
            ["Vanilla DQN", "Dueling DQN"], means, yerr=stds,
            color=[COLOR_DQN, COLOR_DUELING], capsize=8, alpha=0.85,
            edgecolor="black", linewidth=0.8,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate bars
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                    f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Final Performance (last {eval_window} episodes)", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_final_performance.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> Saved: {path}")


# ----------------------------------------------------------------------------- #
#  Plot 4: Stability / Variance                                                 #
# ----------------------------------------------------------------------------- #

def plot_stability(dqn_hist: dict, dueling_hist: dict, output_dir: str):
    """
    Reward curves with shaded ± 1 std rolling variance regions.

    Shows which agent is more consistent during training.

    Args:
        dqn_hist (dict): DQN training history.
        dueling_hist (dict): Dueling DQN training history.
        output_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for hist, color, label in [
        (dqn_hist, COLOR_DQN, "Vanilla DQN"),
        (dueling_hist, COLOR_DUELING, "Dueling DQN"),
    ]:
        eps = np.array(hist["episode"])
        raw = np.array(hist["reward"])
        smoothed = rolling_mean(raw, SMOOTH_WINDOW)
        std = rolling_std(raw, SMOOTH_WINDOW)

        ax.plot(eps, smoothed, color=color, linewidth=2, label=label)
        ax.fill_between(eps, smoothed - std, smoothed + std,
                         color=color, alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Episode Reward")
    ax.set_title("Training Stability (Smoothed Mean ± 1σ)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_stability.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> Saved: {path}")


# ----------------------------------------------------------------------------- #
#  Plot 5: Epsilon Decay Verification                                           #
# ----------------------------------------------------------------------------- #

def plot_epsilon(dqn_hist: dict, dueling_hist: dict, output_dir: str):
    """
    Overlay epsilon (exploration rate) schedules for both agents.

    Verifies that the exploration parameter is decaying correctly.

    Args:
        dqn_hist (dict): DQN training history.
        dueling_hist (dict): Dueling DQN training history.
        output_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for hist, color, label in [
        (dqn_hist, COLOR_DQN, "Vanilla DQN"),
        (dueling_hist, COLOR_DUELING, "Dueling DQN"),
    ]:
        eps = np.array(hist["episode"])
        epsilon = np.array(hist["epsilon"])
        ax.plot(eps, epsilon, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (exploration rate)")
    ax.set_title("Exploration Schedule")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_epsilon.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> Saved: {path}")


# ----------------------------------------------------------------------------- #
#  Main                                                                          #
# ----------------------------------------------------------------------------- #

def main():
    """
    Parse arguments, load histories, and generate all comparison plots.
    """
    parser = argparse.ArgumentParser(
        description="Generate comparative analysis plots (Phase 3, Task 3)"
    )
    parser.add_argument(
        "--dqn-hist", type=str, default="dqn_results/training_history.json",
        help="Path to DQN training history JSON."
    )
    parser.add_argument(
        "--dueling-hist", type=str,
        default="dueling_dqn_results/training_history.json",
        help="Path to Dueling DQN training history JSON."
    )
    parser.add_argument(
        "--output-dir", type=str, default="comparison_plots",
        help="Directory for comparison plots."
    )
    args = parser.parse_args()

    print("Loading training histories ...")
    dqn_hist     = load_history(args.dqn_hist)
    dueling_hist = load_history(args.dueling_hist)
    print(f"  DQN episodes:     {len(dqn_hist['episode'])}")
    print(f"  Dueling episodes: {len(dueling_hist['episode'])}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating comparison plots -> {args.output_dir}/")

    plot_learning_speed(dqn_hist, dueling_hist, args.output_dir)
    plot_loss_convergence(dqn_hist, dueling_hist, args.output_dir)
    plot_final_performance(dqn_hist, dueling_hist, args.output_dir)
    plot_stability(dqn_hist, dueling_hist, args.output_dir)
    plot_epsilon(dqn_hist, dueling_hist, args.output_dir)

    print("\nAll comparison plots generated successfully.")


if __name__ == "__main__":
    main()
