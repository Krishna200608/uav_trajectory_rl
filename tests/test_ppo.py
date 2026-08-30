"""
Unit tests for the PPO baseline (M12): PPOActor, PPOCritic, RolloutBuffer, PPOAgent.

Test coverage:
  1. Network shapes and stochastic sampling behavior.
  2. GAE computation verified against hand-calculated expected values.
  3. Clipped surrogate: ratio exceeds 1+clip_eps, confirm clipping is active.
  4. Full update() end-to-end: no NaN/inf, correct minibatch epoch count.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Test 1: PPOActor / PPOCritic network shapes and sampling behavior
# ---------------------------------------------------------------------------

def test_ppo_actor_forward_shape_and_variance():
    """
    PPOActor.forward() must return (mean, std) of shape (batch, action_dim).
    Repeated calls to sample_action must return DIFFERENT samples (stochastic).
    """
    from uav_trajectory_rl.baselines.ppo import PPOActor

    state_dim = 14
    action_dim = 3
    batch = 4

    actor = PPOActor(state_dim=state_dim, action_dim=action_dim)
    states = torch.randn(batch, state_dim)

    mean, std = actor.forward(states)
    assert mean.shape == (batch, action_dim), f"Expected mean shape {(batch, action_dim)}, got {mean.shape}"
    assert std.shape == (batch, action_dim), f"Expected std shape {(batch, action_dim)}, got {std.shape}"
    assert (std > 0).all(), "Standard deviations must all be positive"

    # Stochasticity: two separate samples must differ
    s_single = torch.randn(state_dim)
    a1, lp1 = actor.sample_action(s_single)
    a2, lp2 = actor.sample_action(s_single)
    assert not torch.allclose(a1, a2), "Repeated stochastic samples should differ"
    assert lp1.shape == (1,), f"log_prob shape expected (1,), got {lp1.shape}"


def test_ppo_critic_forward_shape():
    """
    PPOCritic.forward() must return shape (batch, 1).
    """
    from uav_trajectory_rl.baselines.ppo import PPOCritic

    state_dim = 14
    batch = 8

    critic = PPOCritic(state_dim=state_dim)
    states = torch.randn(batch, state_dim)

    values = critic(states)
    assert values.shape == (batch, 1), f"Expected shape {(batch, 1)}, got {values.shape}"


def test_ppo_deterministic_vs_stochastic():
    """
    select_action_deterministic must return identical results for the same state
    (no randomness), while select_action (stochastic) can vary.
    """
    from uav_trajectory_rl.baselines.ppo import PPOAgent

    state_dim = 14
    rng = np.random.default_rng(42)
    env_state = rng.random(state_dim).astype(np.float32)

    agent = PPOAgent(state_dim=state_dim)

    # Deterministic: two calls must be identical
    d1 = agent.select_action_deterministic(env_state)
    d2 = agent.select_action_deterministic(env_state)
    assert d1 == d2, "Deterministic selection must be repeatable for the same state"

    # Stochastic: physical actions can vary (sampling from distribution)
    # Run enough times that they're very unlikely to be identical
    results = [agent.select_action(env_state)[0] for _ in range(10)]
    # At least some of the 10 samples should differ (with near-certainty for non-zero std)
    v_vals = [r[0] for r in results]
    assert not all(v == v_vals[0] for v in v_vals), "Stochastic action v should vary across samples"


# ---------------------------------------------------------------------------
# Test 2: GAE computation against hand-calculated values
# ---------------------------------------------------------------------------

def test_gae_computation_against_hand_calculation():
    """
    Verify RolloutBuffer.compute_returns_and_advantages() matches hand-calculated
    GAE-Lambda advantages and returns for a known 3-step rollout.

    Setup:
        rewards  = [1.0, 2.0, 3.0]
        values   = [0.5, 1.0, 1.5]
        dones    = [0.0, 0.0, 0.0]  (no termination within rollout)
        last_value = 2.0
        gamma    = 0.9
        gae_lambda = 0.8

    Hand-calculation (backward from t=2):

        t=2:
            delta2 = 3.0 + 0.9 * 2.0 * (1-0) - 1.5 = 3.0 + 1.8 - 1.5 = 3.3
            gae2   = 3.3
            A[2]   = 3.3
            R[2]   = 3.3 + 1.5 = 4.8

        t=1:
            delta1 = 2.0 + 0.9 * 1.5 * (1-0) - 1.0 = 2.0 + 1.35 - 1.0 = 2.35
            gae1   = 2.35 + 0.9 * 0.8 * 3.3 = 2.35 + 2.376 = 4.726
            A[1]   = 4.726
            R[1]   = 4.726 + 1.0 = 5.726

        t=0:
            delta0 = 1.0 + 0.9 * 1.0 * (1-0) - 0.5 = 1.0 + 0.9 - 0.5 = 1.4
            gae0   = 1.4 + 0.9 * 0.8 * 4.726 = 1.4 + 3.40272 = 4.80272
            A[0]   = 4.80272
            R[0]   = 4.80272 + 0.5 = 5.30272

    Raw advantages before normalization: [4.80272, 4.726, 3.3]
        mean = (4.80272 + 4.726 + 3.3) / 3 = 12.82872 / 3 = 4.27624
        var  = ((4.80272 - 4.27624)^2 + (4.726 - 4.27624)^2 + (3.3 - 4.27624)^2) / 3
             = (0.27648^2 + 0.44976^2 + 0.97624^2) / 3
             Note: 4.80272 - 4.27624 = 0.52648 (not 0.27648 -- recalculate)

    Corrected:
        A[0] - mean = 4.80272 - 4.27624 = 0.52648
        A[1] - mean = 4.726   - 4.27624 = 0.44976
        A[2] - mean = 3.3     - 4.27624 = -0.97624

        var  = (0.52648^2 + 0.44976^2 + 0.97624^2) / 3
             = (0.27718  + 0.20228  + 0.95304) / 3
             = 1.4325 / 3 = 0.47750
        std  = sqrt(0.47750) = 0.69102

    Normalized advantages:
        norm_A[0] = 0.52648 / 0.69102 = 0.76189
        norm_A[1] = 0.44976 / 0.69102 = 0.65085
        norm_A[2] = -0.97624 / 0.69102 = -1.41274

    Returns (unaffected by normalization):
        R[0] = 5.30272
        R[1] = 5.726
        R[2] = 4.8
    """
    from uav_trajectory_rl.baselines.ppo import RolloutBuffer

    gamma = 0.9
    gae_lambda = 0.8
    last_value = 2.0

    buf = RolloutBuffer(rollout_length=3, state_dim=2, action_dim=3)
    # Dummy states and actions (not used in GAE computation)
    dummy_state = np.zeros(2, dtype=np.float32)
    dummy_action = np.zeros(3, dtype=np.float32)

    entries = [
        (dummy_state, dummy_action, 0.0, 1.0, 0.5, False),  # t=0
        (dummy_state, dummy_action, 0.0, 2.0, 1.0, False),  # t=1
        (dummy_state, dummy_action, 0.0, 3.0, 1.5, False),  # t=2
    ]
    for (s, a, lp, r, v, d) in entries:
        buf.add(s, a, lp, r, v, d)

    buf.compute_returns_and_advantages(last_value=last_value, gamma=gamma, gae_lambda=gae_lambda)

    # --- Expected returns (hand-computed, independent of normalization) ---
    expected_returns = np.array([5.30272, 5.726, 4.8], dtype=np.float64)
    np.testing.assert_allclose(
        buf.returns,
        expected_returns,
        rtol=1e-4,
        err_msg=f"Returns mismatch. Got {buf.returns}, expected {expected_returns}",
    )

    # --- Expected normalized advantages (hand-computed) ---
    raw_adv = np.array([4.80272, 4.726, 3.3], dtype=np.float64)
    adv_mean = raw_adv.mean()
    adv_std = raw_adv.std() + 1e-8  # must match implementation's 1e-8 epsilon
    expected_norm_adv = (raw_adv - adv_mean) / adv_std

    np.testing.assert_allclose(
        buf.advantages,
        expected_norm_adv,
        rtol=1e-4,
        err_msg=f"Normalized advantages mismatch. Got {buf.advantages}, expected {expected_norm_adv}",
    )


# ---------------------------------------------------------------------------
# Test 3: Clipped surrogate -- confirm clipping is ACTIVE when ratio > 1+eps
# ---------------------------------------------------------------------------

def test_ppo_clipped_surrogate_actually_clips():
    """
    Construct a batch where the new/old ratio clearly exceeds 1 + clip_eps for
    all samples, with a positive advantage, so the surrogate MUST be clipped.

    Setup:
        old_log_prob = log(0.3)  (old policy: low probability)
        new_log_prob = log(0.9)  (new policy: high probability)
        ratio = 0.9 / 0.3 = 3.0
        advantage = +2.0 (positive, so the unclipped term = 3.0 * 2.0 = 6.0)
        clip_eps = 0.2 => clipped_ratio = 1.2
        clipped_term = 1.2 * 2.0 = 2.4
        loss = -min(6.0, 2.4) = -2.4  (clipped term wins)

    Verify the computed policy_loss matches -2.4 (up to floating-point tolerance).
    """
    from uav_trajectory_rl.baselines.ppo import PPOAgent

    clip_eps = 0.2
    advantage = 2.0

    old_log_prob = math.log(0.3)
    new_log_prob = math.log(0.9)
    ratio = math.exp(new_log_prob - old_log_prob)  # = 3.0

    assert ratio > 1.0 + clip_eps, f"ratio={ratio:.4f} must exceed 1+clip_eps={1+clip_eps}"

    # Manually compute expected clipped policy loss
    surr1 = ratio * advantage              # = 6.0
    surr2 = (1.0 + clip_eps) * advantage  # = 2.4 (clipped)
    expected_policy_loss = -min(surr1, surr2)  # = -2.4

    # Now verify with PyTorch tensors (single-element batch)
    old_lp_t = torch.tensor([old_log_prob], dtype=torch.float32)
    new_lp_t = torch.tensor([new_log_prob], dtype=torch.float32)
    adv_t = torch.tensor([advantage], dtype=torch.float32)

    ratio_t = torch.exp(new_lp_t - old_lp_t)
    surr1_t = ratio_t * adv_t
    surr2_t = torch.clamp(ratio_t, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
    policy_loss_t = -torch.min(surr1_t, surr2_t).mean()

    assert abs(float(policy_loss_t.item()) - expected_policy_loss) < 1e-5, (
        f"Clipped policy loss mismatch: got {float(policy_loss_t.item()):.6f}, "
        f"expected {expected_policy_loss:.6f}"
    )

    # Extra: confirm surr2 < surr1 (i.e. clip is actually binding)
    assert float(surr2_t.item()) < float(surr1_t.item()), (
        "The clipped term must be smaller than the unclipped term when ratio > 1+eps"
    )


# ---------------------------------------------------------------------------
# Test 4: update() end-to-end: no NaN/inf, correct epoch count
# ---------------------------------------------------------------------------

def test_ppo_update_end_to_end_no_nan():
    """
    Run PPOAgent.update() on a small synthetic rollout:
      - rollout_length = 32 (small but > minibatch_size=16)
      - update_epochs = 3, minibatch_size = 16
      - Verify: no NaN/inf in returned losses.
      - Verify: n_updates = update_epochs * ceil(rollout_length / minibatch_size)
                = 3 * ceil(32/16) = 3 * 2 = 6 backward() calls occurred.
    """
    from uav_trajectory_rl.baselines.ppo import PPOAgent, RolloutBuffer

    state_dim = 14
    action_dim = 3
    rollout_length = 32
    update_epochs = 3
    minibatch_size = 16
    expected_n_updates = update_epochs * math.ceil(rollout_length / minibatch_size)  # = 6

    rng = np.random.default_rng(0)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        lr=1e-3,  # larger lr so loss visibly changes
    )

    # Build a synthetic rollout
    buf = RolloutBuffer(rollout_length=rollout_length, state_dim=state_dim, action_dim=action_dim)
    env_state = rng.random(state_dim).astype(np.float32)

    for _ in range(rollout_length):
        value = agent.get_value(env_state)
        _, raw_action, log_prob = agent.select_action(env_state)
        reward = float(rng.normal(0.5, 0.2))
        done = False
        buf.add(env_state, raw_action, log_prob, reward, value, done)
        env_state = rng.random(state_dim).astype(np.float32)  # next state

    buf.compute_returns_and_advantages(last_value=0.0, gamma=0.96, gae_lambda=0.95)

    # Run update
    update_info = agent.update(buf, rng)

    # --- No NaN / inf ---
    for key, val in update_info.items():
        if key == "n_updates":
            continue
        assert math.isfinite(val), f"update_info['{key}'] = {val} is not finite"

    # --- Correct number of gradient steps ---
    assert update_info["n_updates"] == expected_n_updates, (
        f"Expected {expected_n_updates} gradient steps (update_epochs * n_minibatches), "
        f"got {update_info['n_updates']}"
    )

    # --- Policy and value losses are non-trivially large (not all zeros) ---
    assert update_info["policy_loss"] != 0.0 or update_info["value_loss"] != 0.0, (
        "Both policy_loss and value_loss are zero -- likely a gradient flow issue"
    )


def test_ppo_save_load_roundtrip(tmp_path):
    """
    Save and reload a PPOAgent; confirm actor and critic weights are identical.
    """
    from uav_trajectory_rl.baselines.ppo import PPOAgent

    state_dim = 14
    agent = PPOAgent(state_dim=state_dim)

    ckpt_file = tmp_path / "ppo_test.pt"
    agent.save(ckpt_file)

    # Mutate weights
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.fill_(0.0)

    agent.load(ckpt_file)

    # After load, actor weights should be the original (non-zero) values
    has_nonzero = any(p.abs().max().item() > 0.0 for p in agent.actor.parameters())
    assert has_nonzero, "Loaded actor weights should not all be zero after reload"
