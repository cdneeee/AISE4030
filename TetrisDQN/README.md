# Tetris DQN – AISE 4030 Reinforcement Learning Project (Phase 3)

A Deep Q-Network (DQN) and Dueling DQN agent trained on a custom Tetris
environment built with the Gymnasium API.  Phase 3 delivers complete
implementations of both algorithms, training runs, and comparative analysis.

---

## Project Structure

```
TetrisDQN/
├── config.yaml              # All hyperparameters and settings (no magic numbers in code)
├── environment.py           # Custom Tetris Gymnasium environment (TetrisEnv)
├── q_network.py             # MLP Q-network architecture (QNetwork)
├── dueling_q_network.py     # Dueling Q-network with Value/Advantage streams
├── replay_buffer.py         # Circular experience replay buffer (ReplayBuffer)
├── dqn_agent.py             # DQN agent – action selection, learning, save/load
├── dueling_dqn_agent.py     # Dueling DQN agent – inherits DQNAgent, overrides network
├── training_script.py       # Main entry point: train, evaluate both agents
├── compare.py               # Comparative analysis: generates all comparison plots
├── utils.py                 # Config loading, device selection, logging, plotting
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── dqn_results/             # Output: DQN checkpoints, training curves, history JSON
├── dueling_dqn_results/     # Output: Dueling DQN checkpoints, curves, history JSON
└── comparison_plots/        # Output: side-by-side comparison figures
```

---

## Setup & Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

### Train DQN (base algorithm)

```bash
python training_script.py --agent dqn
```

### Train Dueling DQN (advanced algorithm)

```bash
python training_script.py --agent dueling
```

### Evaluate a trained model

```bash
# Evaluates over 50 episodes (default) with exploration disabled
python training_script.py --eval --agent dqn
python training_script.py --eval --agent dueling

# Specify checkpoint and episode count
python training_script.py --eval --agent dqn --model dqn_results/dqn_final.pt --eval-episodes 100
```

### Generate comparison plots

```bash
# After both agents have been trained:
python compare.py
```

---

## Reproducibility

All hyperparameters are in `config.yaml`.  To reproduce results:

1. Use Python 3.10+ with the pinned `requirements.txt`.
2. Set `seed: 42` (default) in `config.yaml`.
3. Run `python training_script.py --agent dqn` followed by `--agent dueling`.
4. Run `python compare.py` for all comparison figures.

---

## Configuration

All settings live in `config.yaml`.  Key parameters:

| Parameter | DQN | Dueling DQN | Description |
|---|---|---|---|
| `seed` | 42 | 42 | Global random seed |
| `device` | auto | auto | Compute device: auto / cpu / cuda / mps |
| `num_episodes` | 5000 | 5000 | Training episode count |
| `gamma` | 0.99 | 0.99 | Discount factor |
| `learning_rate` | 0.0001 | 0.0001 | Adam LR |
| `epsilon_start` | 1.0 | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.05 | 0.05 | Minimum exploration rate |
| `epsilon_decay_steps` | 100000 | 100000 | Linear decay horizon |
| `buffer_capacity` | 100000 | 100000 | Replay buffer size |
| `batch_size` | 64 | 64 | Mini-batch size |
| `target_update_freq` | 1000 | 1000 | Target network sync interval |
| `hidden_sizes` | [512, 256, 128] | [512, 256] | Hidden layer widths |
| `value_hidden` | — | 128 | Value stream hidden width |
| `advantage_hidden` | — | 128 | Advantage stream hidden width |

---

## Environment Details

| Property | Value |
|---|---|
| Board size | 20 rows × 10 columns |
| Pieces | 7 standard tetrominoes (I, O, T, S, Z, J, L) |
| Observation shape | `(220,)` float32 in `[0, 1]` |
| Action space | `Discrete(6)` |
| Discount factor (γ) | 0.99 |

**Observation breakdown (220 features):**
- 200 – board occupancy (20×10 binary, row-major)
- 7 – current piece (one-hot)
- 4 – orientation (one-hot, 0–3 clockwise rotations)
- 1 – column position (normalised to [0, 1])
- 1 – row position (normalised to [0, 1])
- 7 – next piece preview (one-hot)

**Actions:**

| Index | Name | Effect |
|---|---|---|
| 0 | LEFT | Move piece one column left |
| 1 | RIGHT | Move piece one column right |
| 2 | ROT_CW | Rotate 90° clockwise |
| 3 | ROT_CCW | Rotate 90° counter-clockwise |
| 4 | DROP | Hard-drop to lowest valid row |
| 5 | NO_OP | No action (gravity applies) |

**Reward function:**

| Event | Reward |
|---|---|
| 1 line cleared | +100 |
| 2 lines cleared | +300 |
| 3 lines cleared | +500 |
| 4 lines cleared (Tetris) | +800 |
| New hole created | −0.5 per hole |
| Height increase | −0.3 per unit |
| Per step | −0.01 |
| Game over | −100 |

---

## Algorithms

### DQN (Base Algorithm)

- **Network**: MLP 220 → 512 → 256 → 128 → 6
- **Target network**: same architecture, weights hard-copied every 1000 gradient steps
- **Loss**: MSE between online Q-values and Bellman targets
- **Exploration**: ε-greedy with linear decay (1.0 → 0.05 over 100k steps)
- **Buffer**: uniform experience replay, capacity 100k

### Dueling DQN (Advanced Algorithm)

- **Shared layers**: 220 → 512 → 256
- **Value stream**: 256 → 128 → 1 (state value V(s))
- **Advantage stream**: 256 → 128 → 6 (per-action advantage A(s,a))
- **Aggregation**: Q(s,a) = V(s) + A(s,a) − mean(A)
- All other components (buffer, exploration, target network, optimizer) are
  identical to the vanilla DQN for a fair controlled comparison.

**Theoretical motivation:** The dueling architecture separates state-value
estimation from action-advantage estimation.  In Tetris, many board states
have similar value regardless of the action chosen (e.g., when the board is
nearly empty).  Dueling DQN can generalise V(s) across actions without
needing to observe every (s, a) pair, improving sample efficiency.

---

## Comparative Analysis (Phase 3, Task 3)

After training both agents, run `python compare.py` to generate:

| Plot | File | Metric |
|---|---|---|
| Learning speed | `comparison_learning_speed.png` | Overlaid reward curves with threshold marker |
| Loss convergence | `comparison_loss_convergence.png` | Overlaid smoothed loss curves |
| Final performance | `comparison_final_performance.png` | Bar chart: mean ± std (last 100 eps) |
| Stability | `comparison_stability.png` | Reward curves with ±1σ shaded region |
| Epsilon schedule | `comparison_epsilon.png` | Exploration decay verification |

---

## Evaluation Metrics

| Metric | Tracking | Rolling window |
|---|---|---|
| Episode reward (primary) | per episode | 100 episodes |
| In-game score | per episode | 100 episodes |
| Lines cleared | per episode | 100 episodes |
| Survival steps | per episode | 100 episodes |
| MSE loss | per episode | 100 episodes |
| Epsilon | per episode | — |

Convergence is defined as the 100-episode rolling score stabilising within
±5 % over 500 consecutive episodes.
