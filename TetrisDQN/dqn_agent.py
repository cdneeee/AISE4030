"""
dqn_agent.py – Deep Q-Network Agent (Baseline Algorithm)
=========================================================
Implements a vanilla DQN agent (Mnih et al., 2015) with:
    • Experience replay buffer for decorrelated updates
    • Target network for stable TD-target computation
    • Epsilon-greedy exploration with linear decay

This is the Phase 2 skeleton.  The class and all public methods are fully
documented; method bodies contain functional implementations where needed
(e.g. action selection, save/load) and clearly marked placeholders where
the learning logic will be completed in Phase 3.

Phase 3 will add a second file, dueling_dqn_agent.py, which inherits from
DQNAgent and overrides only the network architecture (splitting into Value
and Advantage streams) while reusing the replay buffer, training loop, and
exploration schedule unchanged.  This isolates the architectural difference
for a fair comparison.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Vanilla Deep Q-Network agent for discrete-action Tetris control.

    Maintains two Q-networks (online and target), a shared replay buffer,
    and an epsilon-greedy exploration schedule.  The online network is
    updated every step via mini-batch gradient descent; the target network
    receives periodic hard copies of the online weights.

    Hyperparameters are passed via the config dict (loaded from config.yaml)
    to avoid hard-coded magic numbers in the source.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        action_dim (int): Number of discrete actions available.
        config (dict): Agent hyperparameters.  Required keys:
            'learning_rate'       (float) – Adam learning rate.
            'gamma'               (float) – Discount factor.
            'epsilon_start'       (float) – Initial exploration rate.
            'epsilon_end'         (float) – Minimum exploration rate.
            'epsilon_decay_steps' (int)   – Steps to decay epsilon linearly.
            'buffer_capacity'     (int)   – Replay buffer maximum size.
            'batch_size'          (int)   – Mini-batch size for updates.
            'target_update_freq'  (int)   – Gradient steps between target syncs.
            'hidden_sizes'        (list)  – MLP hidden layer widths.
        device (torch.device): Computation device (CPU / CUDA / MPS).
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        config:     dict,
        device:     torch.device,
    ):
        """
        Initialise networks, optimizer, loss function, buffer, and counters.

        Constructs both the online Q-network and a target Q-network with
        identical architecture.  The target network starts with the same
        weights as the online network and is set to eval mode (no gradients).

        Args:
            obs_dim (int): Observation vector dimension.
            action_dim (int): Number of discrete actions.
            config (dict): Hyperparameter dictionary (see class docstring).
            device (torch.device): Torch device for all tensor operations.
        """
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device
        self.config     = config

        # ---- Hyperparameters ----------------------------------------- #
        self.gamma              = float(config["gamma"])
        self.batch_size         = int(config["batch_size"])
        self.target_update_freq = int(config["target_update_freq"])
        self.epsilon            = float(config["epsilon_start"])
        self.epsilon_min        = float(config["epsilon_end"])
        self.epsilon_decay_steps = float(config["epsilon_decay_steps"])

        # ---- Networks ------------------------------------------------- #
        hidden = config.get("hidden_sizes", [512, 256, 128])
        self.online_net = QNetwork(obs_dim, action_dim, hidden).to(device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net never produces gradients

        # ---- Optimiser & loss ----------------------------------------- #
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=float(config["learning_rate"]),
        )
        self.loss_fn = nn.MSELoss()

        # ---- Replay buffer -------------------------------------------- #
        self.buffer = ReplayBuffer(
            capacity=int(config["buffer_capacity"]),
            obs_dim=obs_dim,
            device=device,
        )

        # ---- Training step counter ------------------------------------ #
        self.learn_step = 0  # total gradient update steps performed

    # ------------------------------------------------------------------ #
    #  Action selection                                                    #
    # ------------------------------------------------------------------ #

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using the epsilon-greedy policy.

        During training, a random action is taken with probability epsilon
        (exploration); otherwise the greedy action is chosen (exploitation).
        During evaluation (training=False), epsilon is ignored and the agent
        always acts greedily.

        Args:
            state (np.ndarray): Current state observation, shape (obs_dim,).
            training (bool): If True, apply epsilon-greedy exploration.
                             If False, always choose the greedy action.

        Returns:
            int: Chosen action index in [0, action_dim - 1].
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    #  Experience storage                                                  #
    # ------------------------------------------------------------------ #

    def store_experience(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """
        Store a single (s, a, r, s', done) transition in the replay buffer.

        Args:
            state (np.ndarray): Observation before the action, shape (obs_dim,).
            action (int): Action index taken by the agent.
            reward (float): Scalar reward received.
            next_state (np.ndarray): Resulting observation, shape (obs_dim,).
            done (bool): True if the transition ended the episode.
        """
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------ #
    #  Learning / parameter update                                         #
    # ------------------------------------------------------------------ #

    def learn(self) -> float:
        """
        Sample a mini-batch and perform one gradient descent step.

        Computes the DQN Bellman target:
            y_i = r_i  +  gamma * max_{a'} Q_target(s'_i, a') * (1 - done_i)

        The online network is updated to minimise MSE(Q_online(s,a), y).
        The target network is hard-updated every target_update_freq steps.
        Gradient clipping (max_norm=10) is applied for training stability.

        Returns:
            float: MSE loss for this update step, or 0.0 if the buffer does
                   not yet contain enough transitions (< batch_size).
        """
        if not self.buffer.is_ready(self.batch_size):
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # Q-values of actions actually taken (online network)
        q_current = (
            self.online_net(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        # TD target using the frozen target network
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    # ------------------------------------------------------------------ #
    #  Target network sync                                                 #
    # ------------------------------------------------------------------ #

    def update_target_network(self):
        """
        Hard-copy the online network's weights into the target network.

        Called automatically inside learn() every target_update_freq steps.
        A hard update (rather than Polyak averaging) is used here, consistent
        with the original DQN paper.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------ #
    #  Exploration schedule                                                #
    # ------------------------------------------------------------------ #

    def decay_epsilon(self):
        """
        Linearly decay epsilon towards epsilon_min over epsilon_decay_steps.

        Should be called exactly once per environment step during training.
        After epsilon_decay_steps steps, epsilon stays at epsilon_min for
        the remainder of training.
        """
        step_size = (
            (self.config["epsilon_start"] - self.epsilon_min)
            / self.epsilon_decay_steps
        )
        self.epsilon = max(self.epsilon_min, self.epsilon - step_size)

    # ------------------------------------------------------------------ #
    #  Model persistence                                                   #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str):
        """
        Save network weights, optimizer state, and training counters to disk.

        The checkpoint contains enough information to resume training or
        switch to evaluation mode without retraining.

        Args:
            path (str): Destination file path (e.g. 'dqn_results/dqn_ep500.pt').
                The parent directory is created automatically if needed.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "learn_step": self.learn_step,
            },
            path,
        )

    def load_model(self, path: str):
        """
        Restore network weights, optimizer state, and training counters.

        After loading, target_net is set back to eval mode (no gradients).

        Args:
            path (str): Path to a checkpoint file previously saved by
                save_model().

        Raises:
            FileNotFoundError: If path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon    = checkpoint.get("epsilon",    self.epsilon_min)
        self.learn_step = checkpoint.get("learn_step", 0)
        self.target_net.eval()
