"""
q_network.py – Q-Network Architecture for DQN Tetris Agent
===========================================================
Defines the neural network that maps a flat state observation to Q-values
for every discrete action.  A Multi-Layer Perceptron (MLP) is used because
the state is already a feature vector (not raw pixels), so a CNN is not needed.

Architecture:
    Input(obs_dim=220) → Linear(512) → ReLU
                       → Linear(256) → ReLU
                       → Linear(128) → ReLU
                       → Linear(action_dim=6)

The same class is reused for both the online network and the target network
inside DQNAgent, differing only in whether gradients flow through them.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron that approximates the action-value function Q(s, a).

    Takes a flat state observation vector as input and outputs one Q-value
    per discrete action.  During training, the online copy is updated via
    gradient descent; the target copy has its weights periodically hard-copied
    from the online copy for training stability (Mnih et al., 2015).

    Args:
        obs_dim (int): Dimensionality of the input observation vector.
            Defaults to 220 (standard Tetris state encoding).
        action_dim (int): Number of discrete output actions.
            Defaults to 6 (LEFT, RIGHT, ROT_CW, ROT_CCW, DROP, NO_OP).
        hidden_sizes (list[int], optional): Width of each hidden layer.
            Defaults to [512, 256, 128].
    """

    def __init__(
        self,
        obs_dim: int = 220,
        action_dim: int = 6,
        hidden_sizes: list = None,
    ):
        """
        Initialise the Q-Network layers and apply Xavier weight initialisation.

        Args:
            obs_dim (int): Input feature dimension (default 220).
            action_dim (int): Number of Q-value outputs (default 6).
            hidden_sizes (list[int], optional): Hidden layer sizes.
                Defaults to [512, 256, 128].
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, action_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions given a batch of observations.

        Args:
            x (torch.Tensor): Observation batch of shape (batch, obs_dim).

        Returns:
            torch.Tensor: Q-value tensor of shape (batch, action_dim).
        """
        return self.net(x)

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
