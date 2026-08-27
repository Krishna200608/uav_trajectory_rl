"""
TD3 Actor-Critic Neural Networks and Replay Buffer for PKTD3-TD.

This module implements the neural network architectures and experience replay
buffer corresponding to equations (32)-(38) and Table III of the IEEE TNSE paper:
    "3-D Trajectory Design Based on Deep Reinforcement Learning for
     UAV-Assisted Communication Networks"

Key Components:
    1. Actor:
       Deterministic policy network mapping state s_n -> normalized action in [-c, c]^3.
       Architecture: state_dim -> 256 -> ReLU -> 256 -> ReLU -> 3 -> tanh -> * max_action.
       DESIGN DECISION: Output layer initialized with small uniform weights U(-3e-3, 3e-3)
       (standard DDPG/TD3 literature practice) to keep initial actions centered rather than
       saturated at tanh extremes.
    2. TwinCritic:
       Two independent state-action value networks (Q_theta1, Q_theta2) implementing
       the clipped double-Q learning structure (eq. 32).
       Architecture: (state_dim + action_dim) -> 256 -> ReLU -> 256 -> ReLU -> 1.
    3. ReplayBuffer:
       Fixed-capacity circular buffer storing (s, a, r, s', done) transitions in
       preallocated NumPy arrays for fast sampling without replacement.
    4. to_torch_batch:
       Utility helper converting NumPy array batches into torch.float32 tensors on a target device.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from uav_trajectory_rl.config import ACTION_CLIP_C, HIDDEN_DIM, REPLAY_SIZE

DEFAULT_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    TD3 deterministic actor network mapping state -> normalized action in [-c, c]^3.

    Architecture (Table III):
        Linear(state_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, action_dim) -> tanh -> * max_action
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = HIDDEN_DIM,
        max_action: float = ACTION_CLIP_C,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the Actor network.

        Parameters:
            state_dim: Dimension of the state observation vector (e.g. 2K + 6).
            action_dim: Dimension of the action vector (default: 3 for [v, lam, rho]).
            hidden_dim: Number of neurons per hidden layer (default: 256).
            max_action: Magnitude bound for action clipping c (default: ACTION_CLIP_C = 1.0).
            device: Target torch computation device. Defaults to DEFAULT_DEVICE.
        """
        super().__init__()
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.max_action: float = float(max_action)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(hidden_dim, action_dim)

        # DESIGN DECISION (not specified in the paper):
        # Initialize the final output layer with small uniform weights U(-3e-3, 3e-3).
        # This keeps initial policy outputs near zero / un-saturated rather than pushed
        # against the tanh boundaries (-1 or +1), which stabilizes early TD3 training.
        nn.init.uniform_(self.out_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out_layer.bias, -3e-3, 3e-3)

        target_device = device if device is not None else DEFAULT_DEVICE
        self.to(target_device)
        self.device: torch.device = target_device

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing normalized action in [-max_action, max_action]^3.

        Parameters:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            torch.Tensor: Bounded action tensor of shape (batch_size, action_dim).
        """
        if state.device != self.device:
            state = state.to(self.device)
        features = self.net(state)
        return self.max_action * torch.tanh(self.out_layer(features))


class TwinCritic(nn.Module):
    """
    Twin Q-networks (Q_theta1, Q_theta2) for TD3 clipped double-Q estimation (eq. 32).

    Architecture for each branch (Table III):
        Linear(state_dim + action_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, 1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = HIDDEN_DIM,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the TwinCritic networks.

        Parameters:
            state_dim: Dimension of the state observation vector.
            action_dim: Dimension of the action vector.
            hidden_dim: Number of neurons per hidden layer (default: 256).
            device: Target torch computation device. Defaults to DEFAULT_DEVICE.
        """
        super().__init__()
        in_dim = state_dim + action_dim
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim

        # Q1 architecture branch
        self.q1_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 architecture branch (independent weights)
        self.q2_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q2_out = nn.Linear(hidden_dim, 1)

        # DESIGN DECISION (same as Actor):
        # Initialize output layers with small uniform weights U(-3e-3, 3e-3)
        # to prevent large initial Q-value swings before gradient updates begin.
        nn.init.uniform_(self.q1_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q1_out.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.bias, -3e-3, 3e-3)

        target_device = device if device is not None else DEFAULT_DEVICE
        self.to(target_device)
        self.device: torch.device = target_device

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate both Q-networks on the state-action pair (eq. 32).

        Parameters:
            state: Tensor of shape (batch_size, state_dim).
            action: Tensor of shape (batch_size, action_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Q1, Q2) values, each of shape (batch_size, 1).
        """
        if state.device != self.device:
            state = state.to(self.device)
        if action.device != self.device:
            action = action.to(self.device)
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_out(self.q1_net(sa))
        q2 = self.q2_out(self.q2_net(sa))
        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate ONLY Q1 for the actor policy gradient update (eq. 33).

        Parameters:
            state: Tensor of shape (batch_size, state_dim).
            action: Tensor of shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Q1 values of shape (batch_size, 1).
        """
        if state.device != self.device:
            state = state.to(self.device)
        if action.device != self.device:
            action = action.to(self.device)
        sa = torch.cat([state, action], dim=-1)
        return self.q1_out(self.q1_net(sa))


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing (s, a, r, s', done) transitions.

    Uses preallocated NumPy arrays for high-throughput adding and random sampling.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        capacity: int = REPLAY_SIZE,
    ) -> None:
        """
        Initialize the ReplayBuffer.

        Parameters:
            state_dim: State vector dimension.
            action_dim: Action vector dimension (default: 3).
            capacity: Maximum number of transition tuples stored (default: REPLAY_SIZE = 100,000).
        """
        self.capacity: int = capacity
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards: np.ndarray = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones: np.ndarray = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr: int = 0
        self.size: int = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a new transition tuple into the circular buffer.

        Parameters:
            state: Current state vector.
            action: Action vector taken.
            reward: Scalar reward received.
            next_state: Next state vector observed.
            done: Terminal flag (True if episode ended).
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch of transitions uniformly without replacement.

        Parameters:
            batch_size: Number of transitions to draw.
            rng: NumPy random generator instance.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of current size {self.size}"
            )

        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return self.size


def to_torch_batch(
    *arrays: np.ndarray,
    device: torch.device = DEFAULT_DEVICE,
) -> Tuple[torch.Tensor, ...]:
    """
    Convert a sequence of NumPy arrays to torch.float32 tensors on the given device.

    Parameters:
        *arrays: Variable number of NumPy arrays.
        device: Target computation device.

    Returns:
        Tuple[torch.Tensor, ...]: Converted PyTorch float32 tensors.
    """
    return tuple(torch.as_tensor(arr, dtype=torch.float32, device=device) for arr in arrays)
