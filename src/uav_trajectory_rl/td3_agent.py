"""
TD3 Agent for PKTD3-TD UAV Trajectory Optimization.

This module implements the core Twin Delayed Deep Deterministic Policy Gradient (TD3)
learning algorithm corresponding to equations (32)-(38) and Algorithm 1 (Lines 17-27)
of the IEEE TNSE paper:
    "3-D Trajectory Design Based on Deep Reinforcement Learning for
     UAV-Assisted Communication Networks"

Key Components:
    - Clipped double-Q target evaluation with minimum operator (eq. 32, 36)
    - Target policy smoothing regularization via clamped Gaussian noise (eq. 37-38)
    - Delayed policy and target network updates with frequency `policy_delay` (eq. 33, 35)
    - Polyak parameter soft-updates (eq. 35)
    - Model checkpointing (save / load)
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_trajectory_rl.config import (
    ACTION_CLIP_C,
    GAMMA,
    LEARNING_RATE,
    POLICY_DELAY,
    SIGMA_TILDE,
    TAU,
)
from uav_trajectory_rl.td3_networks import (
    Actor,
    DEFAULT_DEVICE,
    ReplayBuffer,
    TwinCritic,
    to_torch_batch,
)


class TD3Agent:
    """
    TD3 reinforcement learning agent for continuous UAV trajectory control.

    Wraps primary and target Actor and TwinCritic neural networks along with their
    respective Adam optimizers and update logic.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        max_action: float = ACTION_CLIP_C,
        gamma: float = GAMMA,
        tau: float = TAU,
        policy_delay: int = POLICY_DELAY,
        sigma_tilde: float = SIGMA_TILDE,
        action_clip_c: float = ACTION_CLIP_C,
        lr: float = LEARNING_RATE,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the TD3Agent.

        Parameters:
            state_dim: Dimension of state vector s_n (e.g. 2K + 6).
            action_dim: Dimension of action vector a_n (default: 3).
            max_action: Magnitude bound for action clipping (default: ACTION_CLIP_C = 1.0).
            gamma: Discount factor for Bellman updates (default: GAMMA = 0.96).
            tau: Target network soft-update rate (default: TAU = 0.005).
            policy_delay: Actor and target update frequency delay d (default: POLICY_DELAY = 2).
            sigma_tilde: Target policy smoothing noise std dev (default: SIGMA_TILDE = 0.2).
            action_clip_c: Action clipping bound c (default: ACTION_CLIP_C = 1.0).
            lr: Adam optimizer learning rate for actor and critic (default: LEARNING_RATE = 1e-4).
            device: Computation device. Defaults to DEFAULT_DEVICE.
        """
        target_device = device if device is not None else DEFAULT_DEVICE
        self.device: torch.device = target_device

        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.max_action: float = float(max_action)
        self.gamma: float = float(gamma)
        self.tau: float = float(tau)
        self.policy_delay: int = int(policy_delay)
        self.sigma_tilde: float = float(sigma_tilde)
        self.action_clip_c: float = float(action_clip_c)
        self.lr: float = float(lr)

        # Primary and target actor networks
        self.actor = Actor(state_dim, action_dim, max_action=self.max_action, device=self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action=self.max_action, device=self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Primary and target twin critic networks
        self.critic = TwinCritic(state_dim, action_dim, device=self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, device=self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers (Adam, lr=1e-4 from Table III)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Update counter for delayed policy updates
        self.total_updates: int = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Deterministic actor inference mapping state -> normalized action in [-c, c]^3.

        Exploration noise is intentionally omitted here -- that is handled by
        prior_knowledge_policy.select_action in M6, which calls THIS method
        as its actor_fn during the network-driven exploration/exploitation phase.

        Parameters:
            state: NumPy array of shape (state_dim,) or (1, state_dim).

        Returns:
            np.ndarray: Normalized action array of shape (action_dim,) in [-c, c]^3.
        """
        # Note: We skip self.actor.eval()/train() mode switching because the Actor
        # network contains only Linear, ReLU, and tanh layers (no Dropout or
        # BatchNorm), making mode-switching a no-op while avoiding unnecessary overhead.
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            action_t = self.actor(state_t)
        return action_t.squeeze(0).cpu().numpy()

    def _compute_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the clipped double-Q target y (eq. 32, 36-38).

        Target policy smoothing regularization (eq. 37-38):
            a_tilde = clip(mu_theta'(s') + clip(epsilon, -c/2, c/2), -c, c)
            where epsilon ~ N(0, sigma_tilde).

        Clipped double-Q target (eq. 32, 36):
            y = r + gamma * (1 - done) * min(Q_theta1'(s', a_tilde), Q_theta2'(s', a_tilde))
        """
        with torch.no_grad():
            next_action_raw = self.actor_target(next_states)
            noise = torch.randn_like(next_action_raw) * self.sigma_tilde
            # Noise clipping bound in eq. (38): [-c/2, c/2]
            noise = torch.clamp(noise, -self.action_clip_c / 2.0, self.action_clip_c / 2.0)
            next_action = torch.clamp(next_action_raw + noise, -self.action_clip_c, self.action_clip_c)

            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = rewards + self.gamma * (1.0 - dones) * target_q
        return y

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """
        Polyak soft target update (eq. 35):
            theta' = tau * theta + (1 - tau) * theta'
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train_step(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """
        Execute one training step of the TD3 algorithm (Algorithm 1, Lines 17-27).

        Critic networks are updated on every step (eq. 34).
        Actor network and target networks are updated delayed every `policy_delay` steps (eq. 33, 35).

        Parameters:
            replay_buffer: Experience replay buffer storing transition tuples.
            batch_size: Mini-batch size.
            rng: NumPy random generator instance.

        Returns:
            dict: Diagnostics dictionary containing:
                - "critic_loss": Scalar float MSE loss for the twin critic.
                - "actor_loss": Scalar float actor loss if updated, else None.
                - "total_updates": Total number of critic gradient updates performed.
        """
        # 1. Sample mini-batch and convert to torch tensors on self.device
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, rng)
        states, actions, rewards, next_states, dones = to_torch_batch(
            states, actions, rewards, next_states, dones, device=self.device
        )

        # 2. Compute Bellman target y (eq. 32, 36)
        y = self._compute_target(rewards, next_states, dones)

        # 3. Critic loss and update (eq. 34)
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. Increment update counter
        self.total_updates += 1

        # 5. Delayed actor + target network update (eq. 33, 35, Algorithm 1 Line 22)
        if self.total_updates % self.policy_delay == 0:
            # Actor objective: maximize Q1(s, pi(s)) -> minimize -Q1(s, pi(s)) (eq. 33)
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft target updates for actor and twin critic (eq. 35)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            actor_loss_val = float(actor_loss.item())
        else:
            actor_loss_val = None

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_val,
            "total_updates": self.total_updates,
        }

    def save(self, path: str) -> None:
        """Save all model and optimizer state dicts to checkpoint file."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_updates": self.total_updates,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model and optimizer state dicts from checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_updates = int(checkpoint["total_updates"])
