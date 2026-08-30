"""
Dueling Deep Q-Learning (Dueling DQL) baseline for 3D UAV trajectory design (M11).

The IEEE TNSE reference evaluates Dueling DQL [47] as a discrete-action benchmark:
  "Dueling DQL [47]: This method balances UAV's flight time and achieves throughput,
   allowing adjustment of the scaling factor to shorten the UAV's trajectory and
   appropriately increase data collected from users."

Since Dueling DQL is inherently a discrete-action RL method, this module provides:
  1. Action discretization: 3D continuous velocity (v, lam, rho) mapped to 200
     discrete actions (5 speed levels x 5 polar angle levels x 8 azimuth levels).
     (DESIGN DECISION: not specified in the paper; documented as an assumption).
  2. DuelingQNetwork: Shared two-layer MLP trunk (256x256 ReLU) feeding separate
     scalar State-Value V(s) and Action-Advantage A(s, a) streams, combined via:
       Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
  3. DiscreteReplayBuffer: Circular buffer storing scalar discrete actions.
  4. DuelingDQLAgent: Standard DQN learning mechanics with Polyak target updates,
     epsilon-greedy exploration decay, and gradient clipping (max_norm=10.0).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_trajectory_rl.config import (
    GAMMA,
    HIDDEN_DIM,
    LAM_LEVELS,
    LEARNING_RATE,
    NUM_DISCRETE_ACTIONS,
    REPLAY_SIZE,
    RHO_LEVELS,
    TAU,
    V_LEVELS,
)

if TYPE_CHECKING:
    pass


# ==============================================================================
# Action Discretization Helpers (DESIGN DECISION, not paper-specified)
# ==============================================================================

def discrete_action_to_physical(action_idx: int) -> tuple[float, float, float]:
    """
    Map a flat discrete action index in [0, NUM_DISCRETE_ACTIONS - 1] to continuous (v, lam, rho).

    Grid dimensions:
      - v: 5 levels in V_LEVELS (0.0 to 20.0 m/s)
      - lam: 5 levels in LAM_LEVELS (0.0 to pi)
      - rho: 8 levels in RHO_LEVELS (-pi to 3*pi/4)
      Total actions = 5 * 5 * 8 = 200.

    Indexing convention:
      action_idx = v_idx * (len(LAM_LEVELS) * len(RHO_LEVELS)) + lam_idx * len(RHO_LEVELS) + rho_idx
    """
    n_v = len(V_LEVELS)
    n_lam = len(LAM_LEVELS)
    n_rho = len(RHO_LEVELS)
    total = n_v * n_lam * n_rho

    if not 0 <= action_idx < total:
        raise ValueError(f"action_idx {action_idx} is out of bounds [0, {total - 1}].")

    v_idx = action_idx // (n_lam * n_rho)
    rem = action_idx % (n_lam * n_rho)
    lam_idx = rem // n_rho
    rho_idx = rem % n_rho

    return (float(V_LEVELS[v_idx]), float(LAM_LEVELS[lam_idx]), float(RHO_LEVELS[rho_idx]))


def physical_to_nearest_discrete_idx(v: float, lam: float, rho: float) -> int:
    """
    Find the closest discrete action index for a given continuous (v, lam, rho).

    Used for testing, sanity checks, and initializing discrete actions from continuous heuristics.
    Angular distance for azimuth rho accounts for 2*pi circular wrap-around.
    """
    n_lam = len(LAM_LEVELS)
    n_rho = len(RHO_LEVELS)

    # Nearest speed index
    v_idx = int(np.argmin([abs(v - lv) for lv in V_LEVELS]))

    # Nearest polar angle index (clamped in [0, pi])
    lam_clamped = max(0.0, min(math.pi, lam))
    lam_idx = int(np.argmin([abs(lam_clamped - lv) for lv in LAM_LEVELS]))

    # Nearest azimuth index with circular wrap-around in [-pi, pi]
    def angular_dist(a: float, b: float) -> float:
        diff = abs(a - b) % (2.0 * math.pi)
        return min(diff, 2.0 * math.pi - diff)

    rho_idx = int(np.argmin([angular_dist(rho, lv) for lv in RHO_LEVELS]))

    return v_idx * (n_lam * n_rho) + lam_idx * n_rho + rho_idx


# ==============================================================================
# Dueling Q-Network Architecture
# ==============================================================================

class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network (Wang et al., 2016).

    Splits representation after a shared trunk into separate Value and Advantage streams:
      Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))

    Architecture matches the 2-hidden-layer 256-unit MLP used in PKTD3-TD for fairness:
      - Shared trunk: Linear(state_dim, 256) -> ReLU -> Linear(256, 256) -> ReLU
      - Value head: Linear(256, 1)
      - Advantage head: Linear(256, num_actions)
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int = NUM_DISCRETE_ACTIONS,
        hidden_dim: int = HIDDEN_DIM,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device if device is not None else torch.device("cpu")

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate streams
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, num_actions)

        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a) for all discrete actions.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Q-values: Tensor of shape (batch_size, num_actions).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.trunk(state)
        val = self.value_head(features)  # (batch_size, 1)
        adv = self.advantage_head(features)  # (batch_size, num_actions)

        # Identifiability combination: Q = V + (A - mean(A))
        q_vals = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_vals


# ==============================================================================
# Discrete Experience Replay Buffer
# ==============================================================================

class DiscreteReplayBuffer:
    """
    Circular experience replay buffer for discrete action RL.

    Stores scalar int64 actions rather than continuous float action vectors.
    """

    def __init__(self, state_dim: int, capacity: int = REPLAY_SIZE) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a mini-batch of transitions."""
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def to_torch_batch(
        self, batch: tuple, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert sampled NumPy arrays to PyTorch tensors on the target device."""
        states, actions, rewards, next_states, dones = batch
        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return self.size


# ==============================================================================
# Dueling DQL Agent
# ==============================================================================

class DuelingDQLAgent:
    """
    Dueling Deep Q-Learning Agent (M11 baseline).

    Features:
      - Single DuelingQNetwork with target network (no twin critic, no policy delay).
      - Polyak soft target updates executed every training step (tau = 0.005).
      - Epsilon-greedy exploration linearly decaying from epsilon_start to epsilon_end.
      - Gradient clipping (max_norm = 10.0) matching TD3Agent's stability safeguard.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int = NUM_DISCRETE_ACTIONS,
        gamma: float = GAMMA,
        tau: float = TAU,
        lr: float = LEARNING_RATE,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 5000,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.epsilon = epsilon_start

        self.device = device if device is not None else torch.device("cpu")
        self.rng = np.random.default_rng(seed)

        # Primary and target networks
        self.q_net = DuelingQNetwork(state_dim, num_actions, HIDDEN_DIM, self.device)
        self.q_target = DuelingQNetwork(state_dim, num_actions, HIDDEN_DIM, self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def update_epsilon(self, episode: int) -> float:
        """
        Linearly decay epsilon based on the current training episode.

        DESIGN DECISION: Standard linear decay schedule across epsilon_decay_episodes.
        """
        fraction = min(1.0, max(0.0, episode / self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)
        return self.epsilon

    def select_action(self, state: np.ndarray, epsilon: float | None = None) -> int:
        """
        Select a discrete action using epsilon-greedy exploration.

        Args:
            state: Continuous state observation vector, shape (state_dim,).
            epsilon: Optional override for exploration rate (e.g. 0.0 for deterministic eval).
                     If None, uses self.epsilon (current training value).

        Returns:
            Integer action index in [0, num_actions - 1].
        """
        eps = self.epsilon if epsilon is None else epsilon

        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.num_actions))

        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_net(s_tensor)
            return int(torch.argmax(q_vals, dim=-1).item())

    def train_step(
        self,
        replay_buffer: DiscreteReplayBuffer,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """
        Perform a single gradient descent update step on the Dueling Q-network.

        Bellman target:
          y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        Loss:
          MSE(Q(s, a), y)
        """
        batch = replay_buffer.sample(batch_size, rng)
        states, actions, rewards, next_states, dones = replay_buffer.to_torch_batch(batch, self.device)

        with torch.no_grad():
            # Standard DQN target: max_a' Q_target(s', a')
            q_next = self.q_target(next_states)
            max_q_next = q_next.max(dim=1, keepdim=True)[0]
            target_y = rewards + self.gamma * (1.0 - dones) * max_q_next

        # Current Q-values for taken actions
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(q_pred, target_y)

        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update target network on every training step (tau-based Polyak average)
        self._soft_update()

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_pred.mean().item()),
            "mean_target": float(target_y.mean().item()),
        }

    def _soft_update(self) -> None:
        """Polyak average update: theta_target <- tau * theta + (1 - tau) * theta_target."""
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str | Path) -> None:
        """Save model weights and optimizer state."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, filepath: str | Path) -> None:
        """Load model weights and optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
