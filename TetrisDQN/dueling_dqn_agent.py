"""
dueling_dqn_agent.py – Dueling DQN Agent (Advanced Algorithm)
==============================================================
Implements a Dueling DQN agent (Wang et al., 2016) by inheriting from the
vanilla DQNAgent and replacing the Q-network architecture with the Dueling
architecture that separates state-value and action-advantage estimation.

Key difference from vanilla DQN:
    - The Q-network is replaced by a DuelingQNetwork with separate Value
      and Advantage streams, recombined as Q(s,a) = V(s) + A(s,a) - mean(A).
    - All other components (replay buffer, epsilon-greedy exploration,
      training loop, target network updates) remain identical, isolating
      the architectural improvement for a fair comparison.

Theoretical motivation:
    In many states, the choice of action has little effect on the outcome
    (e.g., when the board is nearly empty).  The dueling architecture lets
    the network learn V(s) independently, so it can generalise state value
    across actions without needing to experience every (s, a) pair.  This
    improves sample efficiency and learning speed.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from dqn_agent import DQNAgent
from dueling_q_network import DuelingQNetwork
from replay_buffer import ReplayBuffer


class DuelingDQNAgent(DQNAgent):
    """
    Dueling Deep Q-Network agent for discrete-action Tetris control.

    Inherits all functionality from DQNAgent (experience replay, epsilon-
    greedy exploration, target network sync, save/load).  Overrides only
    the network initialisation to use DuelingQNetwork instead of QNetwork.

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
            'hidden_sizes'        (list)  – Shared feature layer widths.
            'value_hidden'        (int)   – Value stream hidden layer width.
            'advantage_hidden'    (int)   – Advantage stream hidden layer width.
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
        Initialise the Dueling DQN agent with dueling network architecture.

        Constructs DuelingQNetwork instances for both online and target
        networks, replacing the standard QNetwork used in the base class.
        All other components (optimizer, loss, buffer, counters) are
        initialised identically to the vanilla DQN agent.

        Args:
            obs_dim (int): Observation vector dimension.
            action_dim (int): Number of discrete actions.
            config (dict): Hyperparameter dictionary (see class docstring).
            device (torch.device): Torch device for all tensor operations.
        """
        # Skip DQNAgent.__init__ to avoid creating QNetwork instances
        # that would immediately be replaced. Instead, replicate the
        # initialisation with DuelingQNetwork.
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

        # ---- Dueling Networks ----------------------------------------- #
        shared_hidden   = config.get("hidden_sizes", [512, 256])
        value_hidden    = int(config.get("value_hidden", 128))
        adv_hidden      = int(config.get("advantage_hidden", 128))

        self.online_net = DuelingQNetwork(
            obs_dim, action_dim, shared_hidden, value_hidden, adv_hidden
        ).to(device)
        self.target_net = DuelingQNetwork(
            obs_dim, action_dim, shared_hidden, value_hidden, adv_hidden
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

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
        self.learn_step = 0
