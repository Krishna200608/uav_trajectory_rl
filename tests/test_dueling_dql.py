"""
Unit tests for Dueling Deep Q-Learning (Dueling DQL) baseline (M11).

Tests:
  1. discrete_action_to_physical: bounds, known index mappings, out-of-bounds error.
  2. physical_to_nearest_discrete_idx: roundtrip consistency and nearest mapping.
  3. DuelingQNetwork: output shape (batch, 200) and Q-value variance across actions.
  4. DuelingDQLAgent: target network identity at initialization, finite loss, and soft-update divergence.
  5. select_action: deterministic action at epsilon=0.0 vs stochastic exploration at epsilon=1.0.
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch

from uav_trajectory_rl.baselines.dueling_dql import (
    DiscreteReplayBuffer,
    DuelingDQLAgent,
    DuelingQNetwork,
    discrete_action_to_physical,
    physical_to_nearest_discrete_idx,
)
from uav_trajectory_rl.config import (
    LAM_LEVELS,
    NUM_DISCRETE_ACTIONS,
    RHO_LEVELS,
    V_LEVELS,
)


def test_discrete_action_to_physical_bounds_and_known_values():
    """Verify all 200 discrete actions map to physically valid (v, lam, rho) bounds."""
    assert NUM_DISCRETE_ACTIONS == 200

    # Test index 0: first speed, first polar, first azimuth
    v0, lam0, rho0 = discrete_action_to_physical(0)
    assert v0 == V_LEVELS[0] == 0.0
    assert lam0 == LAM_LEVELS[0] == 0.0
    assert rho0 == RHO_LEVELS[0] == -math.pi

    # Test index 199: last speed, last polar, last azimuth
    v_last, lam_last, rho_last = discrete_action_to_physical(199)
    assert v_last == V_LEVELS[-1] == 20.0
    assert lam_last == LAM_LEVELS[-1] == math.pi
    assert rho_last == RHO_LEVELS[-1] == 0.75 * math.pi

    # Test all 200 indices map inside valid kinematic ranges
    for idx in range(200):
        v, lam, rho = discrete_action_to_physical(idx)
        assert 0.0 <= v <= 20.0, f"Speed {v} out of bounds for action {idx}"
        assert 0.0 <= lam <= math.pi, f"Polar angle {lam} out of bounds for action {idx}"
        assert -math.pi <= rho <= math.pi, f"Azimuth angle {rho} out of bounds for action {idx}"

    # Test out-of-bounds indices raise ValueError
    with pytest.raises(ValueError):
        discrete_action_to_physical(-1)
    with pytest.raises(ValueError):
        discrete_action_to_physical(200)


def test_physical_to_nearest_discrete_idx_roundtrip():
    """Verify roundtrip: discrete_action_to_physical -> physical_to_nearest_discrete_idx."""
    for idx in range(NUM_DISCRETE_ACTIONS):
        v, lam, rho = discrete_action_to_physical(idx)
        recovered_idx = physical_to_nearest_discrete_idx(v, lam, rho)
        assert recovered_idx == idx, f"Expected {idx}, got {recovered_idx} for ({v}, {lam}, {rho})"


def test_dueling_q_network_shape_and_variance():
    """Verify DuelingQNetwork output shape and that Q-values are not constant across actions."""
    state_dim = 26
    num_actions = 200
    net = DuelingQNetwork(state_dim=state_dim, num_actions=num_actions)

    batch_size = 4
    dummy_states = torch.randn(batch_size, state_dim)
    q_vals = net(dummy_states)

    assert q_vals.shape == (batch_size, num_actions)

    # Sanity check: Q-values across the 200 actions must have non-zero variance
    # (i.e. the dueling combination formula V + (A - mean(A)) does not collapse to a single constant).
    stds = torch.std(q_vals, dim=-1)
    for b in range(batch_size):
        assert stds[b].item() > 1e-6, f"Q-values are suspiciously identical across actions for batch item {b}"


def test_dueling_dql_agent_construction_and_train_step(tmp_path):
    """
    Verify DuelingDQLAgent:
      1. Target network initializes identical to primary Q-network.
      2. train_step produces finite loss.
      3. Target network updates every train_step via Polyak soft-update (no policy delay).
      4. Save and load preserve weights and state.
    """
    state_dim = 26
    agent = DuelingDQLAgent(state_dim=state_dim, num_actions=200, lr=1e-3, tau=0.05, seed=42)

    # Check identical initialization
    for p_net, p_tgt in zip(agent.q_net.parameters(), agent.q_target.parameters()):
        assert torch.equal(p_net, p_tgt), "Target network was not initialized identical to main network"

    # Fill a mini-buffer with 20 dummy transitions
    rng = np.random.default_rng(42)
    buf = DiscreteReplayBuffer(state_dim=state_dim, capacity=100)
    for _ in range(20):
        s = rng.normal(size=state_dim).astype(np.float32)
        a = int(rng.integers(0, 200))
        r = float(rng.uniform(-2.0, 5.0))
        ns = rng.normal(size=state_dim).astype(np.float32)
        d = bool(rng.random() < 0.1)
        buf.add(s, a, r, ns, d)

    assert len(buf) == 20

    # Save target parameter snapshot before training step
    tgt_weight_before = agent.q_target.trunk[0].weight.clone()

    # Perform train_step
    metrics = agent.train_step(replay_buffer=buf, batch_size=8, rng=rng)

    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["mean_q"])
    assert np.isfinite(metrics["mean_target"])

    # Confirm soft-update immediately modified the target network
    tgt_weight_after = agent.q_target.trunk[0].weight
    assert not torch.equal(tgt_weight_before, tgt_weight_after), (
        "Target network weights did not update after train_step (expected Polyak soft-update on every step)"
    )

    # Check save and load
    save_path = tmp_path / "test_dueling_dql.pt"
    agent.save(save_path)
    assert save_path.exists()

    new_agent = DuelingDQLAgent(state_dim=state_dim, num_actions=200, seed=99)
    new_agent.load(save_path)
    for p1, p2 in zip(agent.q_net.parameters(), new_agent.q_net.parameters()):
        assert torch.equal(p1, p2)


def test_select_action_deterministic_and_exploration():
    """Verify select_action produces deterministic actions when epsilon=0 and exploration when epsilon=1."""
    state_dim = 26
    agent = DuelingDQLAgent(state_dim=state_dim, num_actions=200, seed=123)
    rng = np.random.default_rng(123)
    fixed_state = rng.normal(size=state_dim).astype(np.float32)

    # With epsilon=0.0: repeated queries MUST return the identical greedy action
    greedy_actions = [agent.select_action(fixed_state, epsilon=0.0) for _ in range(25)]
    assert all(a == greedy_actions[0] for a in greedy_actions), "select_action with epsilon=0.0 is not deterministic"

    # With epsilon=1.0: queries MUST produce a wide variety of random actions
    exploratory_actions = [agent.select_action(fixed_state, epsilon=1.0) for _ in range(50)]
    unique_actions = set(exploratory_actions)
    assert len(unique_actions) > 10, f"Expected varied actions with epsilon=1.0, got only {len(unique_actions)} unique"
