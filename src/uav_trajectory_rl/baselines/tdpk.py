"""
Trajectory Design based on Prior Knowledge (TDPK) Baseline (Module M10).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.
    Section V-A (Baseline Description):
    "TDPK employs prior knowledge to guide that the UAV flies directly to the
    destination, and the flight speed for each time slot is randomly generated."

Policy Formulation:
    Unlike M6's prior-knowledge exploration policy (which operates with fixed polar
    angle LAMBDA_PK = 0.5*pi and a randomized azimuth cone [0, 0.5*pi] for training),
    TDPK is a pure geometric direct-flight baseline.
    The flight direction (polar angle lambda, azimuth angle rho) points directly along
    the 3D displacement vector from current_pos to destination:
        delta = destination - current_pos = (dx, dy, dz)
        horizontal_dist = sqrt(dx^2 + dy^2)
        rho = atan2(dy, dx)              # in [-pi, pi]
        lam = atan2(horizontal_dist, dz)  # in [0, pi] from +z axis
    Speed v is uniformly randomized in [0, V_MAX] at each time slot.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from uav_trajectory_rl.config import V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def tdpk_action(
    current_pos: np.ndarray,
    destination: np.ndarray,
    v_max: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Compute the direct-to-destination TDPK action (v, lam, rho).

    Parameters:
        current_pos: 3D coordinates of current UAV position [x, y, z].
        destination: 3D coordinates of target destination [x_end, y_end, z_end].
        v_max: Maximum permissible flight speed (m/s).
        rng: Random number generator for stochastic speed sampling.

    Returns:
        Tuple[float, float, float]: (v, lam, rho) where
            v: Uniform random speed in [0, v_max].
            lam: Polar angle in [0, pi] pointing directly toward destination.
            rho: Azimuth angle in [-pi, pi] pointing directly toward destination.
    """
    dx = float(destination[0] - current_pos[0])
    dy = float(destination[1] - current_pos[1])
    dz = float(destination[2] - current_pos[2])

    horizontal_dist = math.hypot(dx, dy)

    # Edge case: current position matches destination (distance ~ 0)
    # Direction is geometrically undefined; safely return 0.0 for both angles.
    if math.isclose(horizontal_dist, 0.0, abs_tol=1e-9) and math.isclose(dz, 0.0, abs_tol=1e-9):
        lam = 0.0
        rho = 0.0
    else:
        rho = math.atan2(dy, dx)
        lam = math.atan2(horizontal_dist, dz)

    # Randomized speed drawn uniformly from [0, v_max] per paper Sec. V-A
    v = float(rng.uniform(0.0, v_max))

    return (v, lam, rho)


def run_tdpk_episode(
    env: UAVTrajectoryEnv,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run one complete evaluation episode under the TDPK direct-flight policy.

    Parameters:
        env: Instantiated UAVTrajectoryEnv environment.
        rng: NumPy random generator for action randomness.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - episode_reward: float (cumulative 6-term MDP reward)
            - steps_taken: int (number of environment interaction steps)
            - arrived: bool (True if terminal threshold to destination was satisfied)
            - trajectory: np.ndarray of shape (steps_taken + 1, 3) tracking UAV 3D coordinates
    """
    state = env.reset()
    trajectory = [env.uav_pos.copy()]
    episode_reward = 0.0
    done = False
    steps = 0
    arrived = False

    while not done:
        action = tdpk_action(env.uav_pos, env.q_end, V_MAX, rng)
        state, reward, done, info = env.step(action)
        trajectory.append(env.uav_pos.copy())
        episode_reward += reward
        steps += 1
        arrived = info["arrived"]

    return {
        "episode_reward": float(episode_reward),
        "steps_taken": int(steps),
        "arrived": bool(arrived),
        "trajectory": np.array(trajectory, dtype=np.float64),
    }
