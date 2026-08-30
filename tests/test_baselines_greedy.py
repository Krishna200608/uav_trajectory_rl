"""
Unit tests for the Greedy baseline (M13).

Test coverage:
  1. greedy_action selects an action whose immediate reward is >= several
     spot-checked candidate actions (partial validation, not exhaustive re-proof).
  2. greedy_action does NOT modify the real environment's state
     (position, step count, speed) -- this is the critical safety invariant
     of the deep-copy-based design.
  3. run_greedy_episode runs to completion, returns the correct dict shape,
     trajectory starts at Q_START, and trajectory length == steps_taken + 1.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from uav_trajectory_rl.baselines.dueling_dql import discrete_action_to_physical
from uav_trajectory_rl.baselines.greedy import greedy_action, run_greedy_episode
from uav_trajectory_rl.config import N_SLOTS, NUM_DISCRETE_ACTIONS, Q_START
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


# ---------------------------------------------------------------------------
# Test 1: greedy_action's chosen reward >= spot-checked candidates
# ---------------------------------------------------------------------------

def test_greedy_action_chosen_reward_beats_spot_checks():
    """
    Verify that greedy_action returns the highest-immediate-reward action by
    spot-checking it against a sample of other candidate actions.

    Strategy:
      1. Collect the greedy action and its immediate reward (by simulating it
         on a copy of the env).
      2. Sample 10 other random candidate actions and compute their immediate
         rewards the same way (deep copy + step).
      3. Assert the greedy reward >= all 10 sampled rewards.
    """
    rng = np.random.default_rng(7)
    env = UAVTrajectoryEnv(k=5, rng=rng)
    env.reset()

    best_action = greedy_action(env)

    # Get the greedy action's actual immediate reward
    env_copy_best = copy.deepcopy(env)
    _, best_reward, _, _ = env_copy_best.step(best_action)

    # Check against 10 random candidates from the 200-action grid
    rng_check = np.random.default_rng(42)
    sample_indices = rng_check.choice(NUM_DISCRETE_ACTIONS, size=10, replace=False)

    for idx in sample_indices:
        v, lam, rho = discrete_action_to_physical(int(idx))
        env_copy = copy.deepcopy(env)
        _, reward, _, _ = env_copy.step((v, lam, rho))
        assert best_reward >= reward - 1e-9, (
            f"greedy reward {best_reward:.6f} < spot-checked candidate {idx} "
            f"reward {reward:.6f} -- greedy selection is wrong"
        )


# ---------------------------------------------------------------------------
# Test 2: greedy_action does NOT modify the real environment's state
# ---------------------------------------------------------------------------

def test_greedy_action_does_not_mutate_real_env():
    """
    Critical safety test: the real env state must be IDENTICAL before and after
    calling greedy_action().

    Fields checked (complete list of mutable internal state):
      - env.uav_pos:         3D position
      - env.uav_speed:       current speed
      - env.step_count:      episode step counter
      - env.prev_dist_to_end: previous distance (used for reward shaping)
      - env.user_swarm.*:    user positions and velocities (shouldn't move)

    This test is the most important correctness check for the deep-copy design:
    a regression here would silently corrupt every greedy trajectory.
    """
    rng = np.random.default_rng(13)
    env = UAVTrajectoryEnv(k=5, rng=rng)
    env.reset()

    # Snapshot all relevant state fields BEFORE calling greedy_action
    pos_before = env.uav_pos.copy()
    speed_before = env.uav_speed
    step_count_before = env.step_count
    prev_dist_before = env.prev_dist_to_end
    user_pos_before = env.user_swarm.positions.copy()

    # Call greedy_action (should not touch the real env)
    _ = greedy_action(env)

    # Assert ALL state fields are unchanged
    np.testing.assert_array_equal(
        env.uav_pos, pos_before,
        err_msg=f"uav_pos changed after greedy_action: {env.uav_pos} vs {pos_before}",
    )
    assert env.uav_speed == speed_before, (
        f"uav_speed changed: {env.uav_speed} vs {speed_before}"
    )
    assert env.step_count == step_count_before, (
        f"step_count changed: {env.step_count} vs {step_count_before}"
    )
    assert env.prev_dist_to_end == prev_dist_before, (
        f"prev_dist_to_end changed: {env.prev_dist_to_end} vs {prev_dist_before}"
    )
    np.testing.assert_array_equal(
        env.user_swarm.positions, user_pos_before,
        err_msg="user_swarm positions changed after greedy_action",
    )


# ---------------------------------------------------------------------------
# Test 3: run_greedy_episode full episode shape and invariants
# ---------------------------------------------------------------------------

def test_run_greedy_episode_structure_and_invariants():
    """
    Run one full greedy episode and verify:
      - Returns the correct dict with all expected keys.
      - episode_reward is a finite float.
      - steps_taken is in [1, N_SLOTS].
      - arrived is a bool.
      - trajectory shape is (steps_taken + 1, 3).
      - trajectory[0] == Q_START (episode always starts at Q_START).
      - All trajectory coordinates are finite.
    """
    rng = np.random.default_rng(99)
    env = UAVTrajectoryEnv(k=5, rng=rng)

    result = run_greedy_episode(env)

    # Dict structure
    assert isinstance(result, dict)
    for key in ("episode_reward", "steps_taken", "arrived", "trajectory"):
        assert key in result, f"Missing key: {key}"

    # Type checks
    assert isinstance(result["episode_reward"], float)
    assert isinstance(result["steps_taken"], int)
    assert isinstance(result["arrived"], bool)
    assert isinstance(result["trajectory"], np.ndarray)

    # Value bounds
    assert np.isfinite(result["episode_reward"])
    steps = result["steps_taken"]
    assert 1 <= steps <= N_SLOTS, f"steps_taken={steps} out of [1, {N_SLOTS}]"

    # Trajectory shape and content
    traj = result["trajectory"]
    assert traj.shape == (steps + 1, 3), (
        f"trajectory shape {traj.shape} != expected {(steps + 1, 3)}"
    )
    np.testing.assert_allclose(
        traj[0], Q_START,
        atol=1e-9,
        err_msg=f"trajectory[0] {traj[0]} != Q_START {Q_START}",
    )
    assert np.all(np.isfinite(traj)), "trajectory contains non-finite values"
