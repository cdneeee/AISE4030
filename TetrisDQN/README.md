# Tetris DQN – AISE 4030 Reinforcement Learning Project

A Deep Q-Network (DQN) agent trained on a custom Tetris environment built
with the Gymnasium API.  Phase 2 delivers the environment implementation and
the DQN agent skeleton; Phase 3 will complete the training logic and add the
Dueling DQN advanced algorithm for comparison.

---

## Project Structure

```
TetrisDQN/
├── config.yaml            # All hyperparameters and settings (no magic numbers in code)
├── environment.py         # Custom Tetris Gymnasium environment (TetrisEnv)
├── q_network.py           # MLP Q-network architecture (QNetwork)
├── replay_buffer.py       # Circular experience replay buffer (ReplayBuffer)
├── dqn_agent.py           # DQN agent – action selection, learning, save/load (DQNAgent)
├── training_script.py     # Main entry point: train, evaluate, env API confirmation
├── utils.py               # Config loading, device selection, logging, plotting
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── dqn_results/           # Output: checkpoints, training curves, history JSON
```

**Phase 3 additions** (not yet present):
```
dueling_dqn_agent.py       # Dueling DQN — overrides QNetwork with V/A streams
dueling_dqn_results/       # Results for the advanced algorithm
```

---

## Setup & Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify the environment

```bash
cd TetrisDQN
python training_script.py       # prints API info then begins training
```

---

## Running

### Train from scratch

```bash
python training_script.py
python training_script.py --config config.yaml   # explicit config path
```

### Evaluate a saved model

```bash
python training_script.py --eval --model dqn_results/dqn_final.pt
python training_script.py --eval --eval-episodes 20
```

---

## Configuration

All settings live in `config.yaml`.  Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Global random seed |
| `device` | `"auto"` | Compute device: auto / cpu / cuda / mps |
| `num_episodes` | 5000 | Training episode count |
| `agent.gamma` | 0.99 | Discount factor |
| `agent.learning_rate` | 0.0001 | Adam LR |
| `agent.epsilon_start` | 1.0 | Initial exploration rate |
| `agent.epsilon_end` | 0.05 | Minimum exploration rate |
| `agent.epsilon_decay_steps` | 100000 | Linear decay horizon |
| `agent.buffer_capacity` | 100000 | Replay buffer size |
| `agent.batch_size` | 64 | Mini-batch size |
| `agent.target_update_freq` | 1000 | Target network sync interval |

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

## Algorithm: DQN (Baseline)

- **Online network**: MLP 220 → 512 → 256 → 128 → 6
- **Target network**: same architecture, weights copied every 1 000 gradient steps
- **Loss**: MSE between online Q-values and Bellman targets
- **Exploration**: ε-greedy with linear decay (1.0 → 0.05 over 100 000 steps)
- **Buffer**: uniform experience replay, capacity 100 000

**Phase 3 advanced algorithm: Dueling DQN**
The Q-network will be replaced with a dueling architecture that splits the
final layers into a Value stream V(s) and an Advantage stream A(s,a),
combined as Q(s,a) = V(s) + A(s,a) − mean(A).  All other components
(buffer, exploration, target network) remain identical to isolate the
architectural benefit.

---

## Evaluation Metrics (Phase 1 §Evaluation Metric)

| Metric | Tracking | Rolling window |
|---|---|---|
| Episode score (primary) | per episode | 100 episodes |
| Lines cleared (secondary) | per episode | 100 episodes |
| Survival steps (tertiary) | per episode | 100 episodes |

Convergence is defined as the 100-episode rolling score stabilising within
±5 % over 500 consecutive episodes.
