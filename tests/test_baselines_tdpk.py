import math
import numpy as np
import pytest

from uav_trajectory_rl.baselines.tdpk import run_tdpk_episode, tdpk_action
from uav_trajectory_rl.config import N_SLOTS, Q_START, V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def test_tdpk_action_geometry_and_bounds():
    """
    Verify geometric calculations and edge-case behaviors of tdpk_action.
    """
    rng = np.random.default_rng(42)

    # 1. Horizontal diagonal flight (same altitude, dx=600, dy=600, dz=0)
    curr1 = np.array([0.0, 0.0, 50.0])
    dest1 = np.array([600.0, 600.0, 50.0])
    v1, lam1, rho1 = tdpk_action(curr1, dest1, V_MAX, rng)

    assert 0.0 <= v1 <= V_MAX
    # Azimuth should point along xy diagonal: atan2(600, 600) = pi/4
    assert math.isclose(rho1, 0.25 * math.pi, abs_tol=1e-7)
    # Polar angle should be purely horizontal: atan2(horizontal_dist, 0) = pi/2
    assert math.isclose(lam1, 0.5 * math.pi, abs_tol=1e-7)

    # 2. Pure vertical ascent (straight up along +z, dx=0, dy=0, dz=150)
    curr2 = np.array([0.0, 0.0, 50.0])
    dest2 = np.array([0.0, 0.0, 200.0])
    v2, lam2, rho2 = tdpk_action(curr2, dest2, V_MAX, rng)

    assert 0.0 <= v2 <= V_MAX
    # Polar angle should point along +z: lam = 0.0
    assert math.isclose(lam2, 0.0, abs_tol=1e-7)

    # 3. Destination matches current position exactly (edge case)
    curr3 = np.array([300.0, 300.0, 50.0])
    dest3 = np.array([300.0, 300.0, 50.0])
    v3, lam3, rho3 = tdpk_action(curr3, dest3, V_MAX, rng)

    assert 0.0 <= v3 <= V_MAX
    assert lam3 == 0.0
    assert rho3 == 0.0

    # 4. Pure vertical descent (straight down along -z, dx=0, dy=0, dz=-50)
    curr4 = np.array([100.0, 100.0, 100.0])
    dest4 = np.array([100.0, 100.0, 50.0])
    v4, lam4, rho4 = tdpk_action(curr4, dest4, V_MAX, rng)

    assert 0.0 <= v4 <= V_MAX
    # Polar angle straight down: lam = pi
    assert math.isclose(lam4, math.pi, abs_tol=1e-7)

    # 5. Stochastic speed bound verification across 1,000 draws
    speeds = [tdpk_action(curr1, dest1, V_MAX, rng)[0] for _ in range(1000)]
    assert min(speeds) >= 0.0
    assert max(speeds) <= V_MAX
    assert 0.45 * V_MAX <= np.mean(speeds) <= 0.55 * V_MAX


def test_run_tdpk_episode_full():
    """
    Run one full TDPK episode with seeded environment and verify episode metrics.
    """
    rng = np.random.default_rng(123)
    env = UAVTrajectoryEnv(k=5, rng=rng)

    result = run_tdpk_episode(env, rng)

    assert isinstance(result, dict)
    assert "episode_reward" in result
    assert "steps_taken" in result
    assert "arrived" in result
    assert "trajectory" in result

    steps = result["steps_taken"]
    assert 1 <= steps <= N_SLOTS
    assert isinstance(result["arrived"], bool)

    # Reward must be finite float
    assert isinstance(result["episode_reward"], float)
    assert np.isfinite(result["episode_reward"])

    # Trajectory shape must be (steps + 1, 3) and start at Q_START
    traj = result["trajectory"]
    assert traj.shape == (steps + 1, 3)
    assert np.allclose(traj[0], Q_START)

    # All coordinates must be finite
    assert np.all(np.isfinite(traj))
