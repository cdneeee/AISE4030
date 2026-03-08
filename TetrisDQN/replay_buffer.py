"""
replay_buffer.py – Experience Replay Buffer for DQN
====================================================
Stores (state, action, reward, next_state, done) transition tuples and
provides uniform random sampling for off-policy TD learning.

The buffer is implemented as a fixed-size circular array using pre-allocated
NumPy arrays, which is significantly more memory-efficient than a deque of
Python tuples.  Tensors are created only at sample time.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular buffer for storing and sampling RL transitions.

    Transitions are stored as contiguous NumPy arrays and converted to
    PyTorch tensors on demand.  Once capacity is reached, the oldest
    experience is overwritten (FIFO eviction).

    Args:
        capacity (int): Maximum number of transitions the buffer can hold.
        obs_dim (int): Dimensionality of each observation vector.
        device (torch.device): Device onto which sampled tensors are placed.
    """

    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        """
        Allocate pre-sized NumPy arrays for all transition fields.

        Args:
            capacity (int): Maximum buffer size (number of transitions).
            obs_dim (int): Observation space dimensionality.
            device (torch.device): Torch device for sampled output tensors.
        """
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.device   = device

        self._size = 0   # current number of valid transitions
        self._ptr  = 0   # write pointer (wraps around)

        # Pre-allocated storage arrays
        self._states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity,            dtype=np.int64)
        self._rewards     = np.zeros(capacity,            dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones       = np.zeros(capacity,            dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Storage                                                             #
    # ------------------------------------------------------------------ #

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """
        Add a single transition to the buffer.

        If the buffer is full the oldest transition is overwritten.

        Args:
            state (np.ndarray): Observation before the action, shape (obs_dim,).
            action (int): Discrete action index taken by the agent.
            reward (float): Scalar reward received after taking the action.
            next_state (np.ndarray): Resulting observation, shape (obs_dim,).
            done (bool): True if the transition ended the episode.
        """
        self._states[self._ptr]      = state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr]     = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr]       = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------ #
    #  Sampling                                                            #
    # ------------------------------------------------------------------ #

    def sample(self, batch_size: int):
        """
        Uniformly sample a mini-batch of transitions without replacement.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple of torch.Tensor:
                states      (torch.Tensor): shape (batch_size, obs_dim), float32
                actions     (torch.Tensor): shape (batch_size,),         int64
                rewards     (torch.Tensor): shape (batch_size,),         float32
                next_states (torch.Tensor): shape (batch_size, obs_dim), float32
                dones       (torch.Tensor): shape (batch_size,),         float32

        Raises:
            ValueError: If batch_size exceeds the current number of stored
                        transitions (check __len__ before calling).
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer "
                f"with only {self._size} stored."
            )
        idx = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.tensor(self._states[idx],      dtype=torch.float32).to(self.device),
            torch.tensor(self._actions[idx],     dtype=torch.long).to(self.device),
            torch.tensor(self._rewards[idx],     dtype=torch.float32).to(self.device),
            torch.tensor(self._next_states[idx], dtype=torch.float32).to(self.device),
            torch.tensor(self._dones[idx],       dtype=torch.float32).to(self.device),
        )

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """
        Return the current number of stored transitions.

        Returns:
            int: Number of valid transitions in the buffer (≤ capacity).
        """
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        """
        Check whether the buffer holds enough transitions to yield a batch.

        Args:
            batch_size (int): Required minimum number of transitions.

        Returns:
            bool: True if len(self) >= batch_size.
        """
        return self._size >= batch_size
