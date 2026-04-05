"""
dueling_q_network.py – Dueling Q-Network Architecture
======================================================
Implements the Dueling Network Architecture (Wang et al., 2016) for the
Tetris DQN agent.  Instead of directly estimating Q(s, a), the network
splits into two streams after a shared feature extractor:

    1. Value stream   V(s)   – scalar estimate of state value.
    2. Advantage stream A(s, a) – per-action advantage over the mean.

Q-values are reconstructed via:
    Q(s, a) = V(s) + A(s, a) - mean_a[ A(s, a) ]

Subtracting the mean advantage ensures identifiability: without it,
V and A are not uniquely determined from Q alone.

Architecture:
    Input(obs_dim=220)
        → Linear(512) → ReLU
        → Linear(256) → ReLU       [shared feature layers]
        ┌─────────────────────────────────────────────┐
        │ Value stream:      Linear(128) → ReLU → Linear(1)          │
        │ Advantage stream:  Linear(128) → ReLU → Linear(action_dim) │
        └─────────────────────────────────────────────┘
        → Q(s,a) = V(s) + A(s,a) - mean(A)
"""

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network that decomposes Q-values into Value and Advantage.

    Shares initial feature extraction layers with the standard QNetwork,
    then branches into separate value and advantage streams.  This
    architecture allows the network to learn the general value of a state
    independently from the relative advantage of each action, improving
    learning efficiency especially in states where action choice matters
    less than being in the right state.

    Args:
        obs_dim (int): Dimensionality of the input observation vector.
            Defaults to 220 (standard Tetris state encoding).
        action_dim (int): Number of discrete output actions.
            Defaults to 6 (LEFT, RIGHT, ROT_CW, ROT_CCW, DROP, NO_OP).
        hidden_sizes (list[int], optional): Width of each shared feature
            layer. Defaults to [512, 256].
        value_hidden (int, optional): Width of the value stream hidden
            layer. Defaults to 128.
        advantage_hidden (int, optional): Width of the advantage stream
            hidden layer. Defaults to 128.
    """

    def __init__(
        self,
        obs_dim: int = 220,
        action_dim: int = 6,
        hidden_sizes: list = None,
        value_hidden: int = 128,
        advantage_hidden: int = 128,
    ):
        """
        Initialise shared feature layers, value stream, and advantage stream.

        Args:
            obs_dim (int): Input feature dimension (default 220).
            action_dim (int): Number of Q-value outputs (default 6).
            hidden_sizes (list[int], optional): Shared hidden layer sizes.
                Defaults to [512, 256].
            value_hidden (int): Hidden units in the value stream (default 128).
            advantage_hidden (int): Hidden units in the advantage stream
                (default 128).
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        # ---- Shared feature extractor --------------------------------- #
        shared_layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            shared_layers.append(nn.Linear(in_size, h))
            shared_layers.append(nn.ReLU())
            in_size = h
        self.shared = nn.Sequential(*shared_layers)

        # ---- Value stream: V(s) → scalar ------------------------------ #
        self.value_stream = nn.Sequential(
            nn.Linear(in_size, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

        # ---- Advantage stream: A(s, a) → per-action ------------------- #
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_size, advantage_hidden),
            nn.ReLU(),
            nn.Linear(advantage_hidden, action_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values by combining value and advantage streams.

        Q(s, a) = V(s) + A(s, a) - mean_a[A(s, a)]

        Args:
            x (torch.Tensor): Observation batch of shape (batch, obs_dim).

        Returns:
            torch.Tensor: Q-value tensor of shape (batch, action_dim).
        """
        features = self.shared(x)
        value = self.value_stream(features)             # (batch, 1)
        advantage = self.advantage_stream(features)     # (batch, action_dim)
        # Subtract mean advantage for identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    # ------------------------------------------------------------------ #
    #  Weight initialisation                                               #
    # ------------------------------------------------------------------ #

    def _init_weights(self):
        """
        Apply Xavier uniform initialisation to all Linear layers.

        Xavier initialisation keeps the variance of activations consistent
        across layers, which accelerates convergence in deep networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
