"""
Proximal Policy Optimization (PPO) baseline for 3D UAV trajectory design (M12).

The IEEE TNSE reference evaluates PPO [44] as a continuous-action benchmark:
  "PPO [44]: The PPO algorithm is used for UAV 3D trajectory planning with the
   same state and reward settings as the proposed method. At each time slot,
   the agent generates continuous control actions to update the UAV's flight
   direction and speed based on the learned policy."

Unlike Dueling DQL (M11), PPO operates in the SAME continuous action space as
PKTD3-TD -- no discretization. The normalized [-c, c]^3 action convention and
unnormalize_action mapping are reused unchanged.

Reference [44] (Schulman et al., 2017) does not specify architecture or
hyperparameters for this domain. All design decisions are explicitly documented.

Design decisions:
  1. Separate actor (PPOActor) and critic (PPOCritic) networks -- no shared trunk.
     Rationale: Avoids interference between policy gradient and value regression
     objectives on the same representation; simpler to tune.
  2. State-independent log_std: A single learnable nn.Parameter of shape
     (action_dim,), initialized to log(0.5), shared across all states.
     Rationale: Standard OpenAI Baselines / SpinningUp practice. Simpler than
     state-dependent std, which can cause log-prob instability early in training.
  3. Unsquashed Gaussian mean output (plain Linear, no tanh):
     Rationale: Avoids tanh log-prob correction complexity. Sampled actions are
     clipped to [-c, c] at execution, but log-probs are computed on unclipped
     samples. Known mild bias: the log-prob does not account for the clipping
     probability mass. This is an accepted simplification (paper does not specify
     at this level of detail).
  4. GAE-Lambda with lambda=0.95 (not paper-specified).
  5. Rollout length: 2048 steps (standard PPO default, not paper-specified).
  6. PPO-Clip with epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
     update_epochs=10, minibatch_size=64 (standard PPO defaults).
  7. Combined optimizer over both actor and critic parameters
     (single Adam call). Rationale: Simpler code, equivalent to separate
     optimizers with the same LR, which is how LR is set here.
  8. Gradient clipping: max_norm=10.0, matching TD3Agent and DuelingDQLAgent.
  9. Learning rate: 1e-4 (reused from config LEARNING_RATE for consistency
     with every other module in this project; PPO implementations often use
     3e-4, but consistency across baselines is preferred here).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_trajectory_rl.config import (
    ACTION_CLIP_C,
    GAMMA,
    HIDDEN_DIM,
    LEARNING_RATE,
    V_MAX,
)
from uav_trajectory_rl.prior_knowledge_policy import unnormalize_action


# ==============================================================================
# Actor: Gaussian Policy Network
# ==============================================================================


class PPOActor(nn.Module):
    """
    Gaussian policy network for PPO (M12 baseline).

    Outputs a Normal distribution over the normalized [-c, c]^3 action space.

    Architecture (DESIGN DECISION: matches 2x256 ReLU trunk used in PKTD3-TD):
        state_dim -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> mean (Linear, action_dim)

    The standard deviation is parameterized by a state-INDEPENDENT learnable
    log_std vector of shape (action_dim,), initialized to log(0.5) so that the
    initial std=0.5 gives moderate exploration over the [-1, 1]^3 normalized space.

    Known limitation: log-probs are computed on unclipped Gaussian samples
    (clipping to [-c, c] happens at the caller level). This introduces a mild
    bias in the PPO ratio because the effective distribution is a truncated
    Gaussian, not a full Gaussian. The paper does not specify at this level of
    detail; this is an accepted simplification documented here explicitly.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = HIDDEN_DIM,
        log_std_init: float = math.log(0.5),
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        # MLP trunk producing the mean action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # State-independent log_std (DESIGN DECISION -- see module docstring)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init, dtype=torch.float32)
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Gaussian mean and std for a batch of states.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            mean: Tensor of shape (batch_size, action_dim).
            std:  Tensor of shape (batch_size, action_dim).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Return a Normal distribution object for the given state(s)."""
        mean, std = self.forward(state)
        return torch.distributions.Normal(mean, std)

    def sample_action(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy and compute its log-probability.

        The returned action is UNCLIPPED (raw Gaussian sample). Clipping to
        [-c, c] and conversion to physical units happen at the caller level.
        Log-prob is computed on the unclipped sample (see module docstring for
        known-limitation note).

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            action:   Tensor of shape (batch_size, action_dim) -- raw sample.
            log_prob: Tensor of shape (batch_size,) -- sum of component log-probs.
        """
        dist = self.get_distribution(state)
        action = dist.rsample()  # reparameterized for differentiability
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        return action, log_prob


# ==============================================================================
# Critic: State-Value Network
# ==============================================================================


class PPOCritic(nn.Module):
    """
    State-value network V(s) for PPO advantage estimation.

    NOTE: PPO's critic takes only the state as input (no action), unlike
    TD3's Q(s, a) critic. This is architecturally distinct and correct.

    Architecture (DESIGN DECISION: same 2x256 ReLU hidden layers as actor):
        state_dim -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(1)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate V(s).

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            values: Tensor of shape (batch_size, 1).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)


# ==============================================================================
# Rollout Buffer with GAE
# ==============================================================================


class RolloutBuffer:
    """
    Fixed-length on-policy rollout buffer with Generalized Advantage Estimation
    (GAE-Lambda, Schulman et al., 2016).

    Unlike TD3's circular replay buffer, this buffer:
      - Stores exactly rollout_length transitions.
      - Is fully consumed and then discarded after each PPO update cycle.
      - Never mixes transitions from different update epochs.

    GAE-Lambda (DESIGN DECISION: lambda=0.95, standard default):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        returns_t = A_t + V(s_t)

    Advantages are normalized (zero mean, unit std) across the rollout batch
    after computation -- standard PPO training stability practice.
    """

    def __init__(
        self,
        rollout_length: int,
        state_dim: int,
        action_dim: int = 3,
    ) -> None:
        self.rollout_length = rollout_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.full = False

        # Preallocate storage arrays
        self.states = np.zeros((rollout_length, state_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)

        # Computed by compute_returns_and_advantages
        self.returns = np.zeros(rollout_length, dtype=np.float32)
        self.advantages = np.zeros(rollout_length, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store a single step's data at the current write pointer."""
        assert self.ptr < self.rollout_length, (
            "RolloutBuffer is full. Call reset() before adding more transitions."
        )
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = float(log_prob)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.ptr += 1
        if self.ptr == self.rollout_length:
            self.full = True

    def reset(self) -> None:
        """Clear the buffer for a new rollout collection cycle."""
        self.ptr = 0
        self.full = False

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute GAE-Lambda advantages and discounted returns in-place.

        This method is called once after the rollout is complete, before
        calling update(). It computes backward from the end of the rollout.

        Args:
            last_value: V(s_{T+1}), the bootstrapped value of the state
                        immediately AFTER the last stored step. Should be
                        set to 0.0 if the last step is a true episode
                        termination (done=True), or to the critic's estimate
                        of the following state if the rollout was truncated
                        mid-episode.
            gamma:      Discount factor.
            gae_lambda: GAE smoothing parameter (DESIGN DECISION: 0.95).
        """
        n = self.rollout_length
        gae = 0.0

        for t in reversed(range(n)):
            # Bootstrap V(s_{t+1}): use last_value for the final step,
            # otherwise use the stored value of the next step.
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            not_done = 1.0 - self.dones[t]

            # Temporal difference error
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]

            # Recursive GAE accumulation (backward)
            gae = delta + gamma * gae_lambda * not_done * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

        # Normalize advantages across the whole rollout batch for training stability
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_minibatches(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Iterator[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Yield shuffled minibatches of (states, actions, old_log_probs, returns, advantages).

        Args:
            batch_size: Minibatch size.
            rng:        NumPy random generator for reproducible shuffling.

        Yields:
            Tuples of (states, actions, old_log_probs, returns, advantages),
            each of shape (B, ...) where B <= batch_size.
        """
        indices = rng.permutation(self.rollout_length)
        start = 0
        while start < self.rollout_length:
            end = min(start + batch_size, self.rollout_length)
            idx = indices[start:end]
            yield (
                self.states[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.returns[idx],
                self.advantages[idx],
            )
            start = end


# ==============================================================================
# PPO Agent
# ==============================================================================


class PPOAgent:
    """
    Proximal Policy Optimization Agent (M12 baseline).

    Implements PPO-Clip (Schulman et al., 2017) with:
      - Gaussian stochastic policy (PPOActor)
      - State-value baseline (PPOCritic)
      - GAE-Lambda advantage estimation (RolloutBuffer)
      - Clipped surrogate objective with entropy bonus
      - Combined single Adam optimizer over actor + critic parameters
        (DESIGN DECISION: logistically simpler than two separate optimizers;
         equivalent when the same LR is used for both, which is the case here)

    Action interface:
      - Training:    select_action() -> (physical_action, raw_normalized, log_prob)
      - Evaluation:  select_action_deterministic() -> physical_action (mean only)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        gamma: float = GAMMA,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        lr: float = LEARNING_RATE,
        max_grad_norm: float = 10.0,
        device: torch.device | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.c = ACTION_CLIP_C  # action clipping bound in normalized space

        self.device = device if device is not None else torch.device("cpu")

        # Separate networks (DESIGN DECISION -- see module docstring)
        self.actor = PPOActor(state_dim, action_dim, HIDDEN_DIM).to(self.device)
        self.critic = PPOCritic(state_dim, HIDDEN_DIM).to(self.device)

        # Single combined optimizer over both networks' parameters
        # (DESIGN DECISION: same lr=1e-4 for both, consistent with PKTD3-TD)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    # --------------------------------------------------------------------------
    # Action Selection
    # --------------------------------------------------------------------------

    def select_action(
        self, state: np.ndarray
    ) -> tuple[tuple[float, float, float], np.ndarray, float]:
        """
        Sample an action from the stochastic policy for rollout collection.

        Steps:
          1. Convert state to tensor, sample (raw_action, log_prob) from actor.
          2. Clip raw_action to [-c, c]^3 in normalized space.
          3. Convert to physical (v, lam, rho) via unnormalize_action.

        Args:
            state: Numpy state vector of shape (state_dim,).

        Returns:
            physical_action: Tuple (v, lam, rho) for env.step().
            raw_clipped:     Numpy array of shape (action_dim,) in [-c, c]^3.
            log_prob:        Float log-probability of the raw UNCLIPPED sample
                             (see module docstring for known-limitation note).
        """
        s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            raw_action, log_prob = self.actor.sample_action(s_tensor)

        raw_np = raw_action.squeeze(0).cpu().numpy()
        raw_clipped = np.clip(raw_np, -self.c, self.c)
        physical_action = unnormalize_action(raw_clipped, c=self.c)

        return physical_action, raw_np, float(log_prob.item())

    def get_value(self, state: np.ndarray) -> float:
        """Evaluate V(s) for a single state (used during rollout collection)."""
        s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.critic(s_tensor)
        return float(value.item())

    def evaluate_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probabilities and entropy under the CURRENT policy.

        Called during each PPO update epoch to evaluate the old rollout actions
        under the new (updated) policy weights. This is what gives PPO the
        "new/old ratio" in the clipped surrogate objective.

        Args:
            states:  Tensor of shape (batch_size, state_dim).
            actions: Tensor of shape (batch_size, action_dim) -- raw, unclipped.

        Returns:
            log_probs: Tensor of shape (batch_size,).
            entropy:   Tensor of shape (batch_size,) -- sum over action dims.
        """
        dist = self.actor.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

    def select_action_deterministic(self, state: np.ndarray) -> tuple[float, float, float]:
        """
        Deterministic evaluation: use the Gaussian MEAN directly (no sampling).

        Matches the "deterministic evaluation" convention used for every other
        method in this project (epsilon=0 for Dueling DQL; actor mean for TD3).

        Args:
            state: Numpy state vector of shape (state_dim,).

        Returns:
            physical_action: Tuple (v, lam, rho) for env.step().
        """
        s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, _ = self.actor(s_tensor)
        mean_np = mean.squeeze(0).cpu().numpy()
        mean_clipped = np.clip(mean_np, -self.c, self.c)
        return unnormalize_action(mean_clipped, c=self.c)

    # --------------------------------------------------------------------------
    # Policy Update
    # --------------------------------------------------------------------------

    def update(
        self,
        rollout_buffer: RolloutBuffer,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """
        Perform update_epochs passes over the rollout batch using PPO-Clip.

        PPO-Clip objective (Schulman et al., 2017):
            ratio    = exp(log_pi_new(a|s) - log_pi_old(a|s))
            surr1    = ratio * A_t
            surr2    = clip(ratio, 1 - clip_eps, 1 + clip_eps) * A_t
            L_policy = -min(surr1, surr2)                   [maximize = minimize negative]
            L_value  = MSE(V(s), returns)
            L_entropy= entropy.mean()                        [maximize = minimize negative]
            L_total  = L_policy + value_coef * L_value - entropy_coef * L_entropy

        Multiple epochs over the SAME rollout batch is what distinguishes PPO
        from single-pass on-policy methods and provides sample efficiency.

        Args:
            rollout_buffer: Completed RolloutBuffer with computed advantages/returns.
            rng:            NumPy generator for minibatch shuffling.

        Returns:
            Dictionary with mean 'policy_loss', 'value_loss', 'entropy' across
            all minibatches and epochs, for logging.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _epoch in range(self.update_epochs):
            for batch in rollout_buffer.get_minibatches(self.minibatch_size, rng):
                states_np, actions_np, old_log_probs_np, returns_np, advantages_np = batch

                # Convert to tensors
                states_t = torch.tensor(states_np, dtype=torch.float32, device=self.device)
                actions_t = torch.tensor(actions_np, dtype=torch.float32, device=self.device)
                old_log_probs_t = torch.tensor(old_log_probs_np, dtype=torch.float32, device=self.device)
                returns_t = torch.tensor(returns_np, dtype=torch.float32, device=self.device)
                advantages_t = torch.tensor(advantages_np, dtype=torch.float32, device=self.device)

                # Current policy evaluation
                new_log_probs_t, entropy_t = self.evaluate_action(states_t, actions_t)

                # Clipped surrogate policy loss
                ratio = torch.exp(new_log_probs_t - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss: MSE between critic prediction and GAE returns
                values_pred = self.critic(states_t).squeeze(-1)
                value_loss = F.mse_loss(values_pred, returns_t)

                # Entropy bonus (maximize entropy => subtract from total loss)
                entropy_mean = entropy_t.mean()

                # Combined loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                # Gradient step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_mean.item())
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_updates": n_updates,
        }

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------

    def save(self, filepath: str | Path) -> None:
        """Save actor, critic, and optimizer state dicts."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, filepath: str | Path) -> None:
        """Load actor, critic, and optimizer state dicts."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
