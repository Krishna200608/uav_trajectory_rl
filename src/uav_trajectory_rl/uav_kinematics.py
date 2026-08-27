"""
UAV Kinematics and Spherical Coordinate Motion Model.

This module models the 3D kinematic motion of an Unmanned Aerial Vehicle (UAV)
using spherical coordinates (v, lambda, rho):
    - v: Flight speed in [0, V_MAX] (m/s)
    - lambda (polar angle): Angle with positive z-axis in [0, pi] (rad)
    - rho (azimuth angle): Angle in xy-plane from positive x-axis in [-pi, pi] (rad)

The velocity vector direction aligns with the acceleration direction.
Position updates in discrete time slots are given by:
    x_{n+1} = x_n + v_n * sin(lambda_n) * cos(rho_n) * delta
    y_{n+1} = y_n + v_n * sin(lambda_n) * sin(rho_n) * delta
    z_{n+1} = z_n + v_n * cos(lambda_n) * delta
"""

import math
from typing import Sequence, Union
import numpy as np

from uav_trajectory_rl.config import V_MAX, AC_MAX, DELTA


def step_uav_position(
    position: Union[Sequence[float], np.ndarray],
    v: float,
    lam: float,
    rho: float,
    delta: float = DELTA,
) -> np.ndarray:
    """
    Compute the UAV's next 3D position given current position and spherical action.

    Parameters:
        position: Current 3D position [x_n, y_n, z_n] in meters.
        v: Flight speed v_n in [0, V_MAX] (m/s).
        lam: Polar angle lambda_n in [0, pi] (radians).
        rho: Azimuth angle rho_n in [-pi, pi] (radians).
        delta: Time slot duration in seconds (default from config: DELTA).

    Returns:
        np.ndarray: Updated 3D position [x_{n+1}, y_{n+1}, z_{n+1}] in meters.
    """
    curr_x, curr_y, curr_z = position[0], position[1], position[2]

    # Spherical coordinate displacement decomposition
    sin_lam = math.sin(lam)
    dx = v * sin_lam * math.cos(rho) * delta
    dy = v * sin_lam * math.sin(rho) * delta
    dz = v * math.cos(lam) * delta

    next_x = curr_x + dx
    next_y = curr_y + dy
    next_z = curr_z + dz

    return np.array([next_x, next_y, next_z], dtype=np.float64)


def apply_acceleration_constraint(
    v_prev: float,
    requested_v: float,
    ac_max: float = AC_MAX,
    delta: float = DELTA,
) -> tuple[float, bool]:
    """
    Apply acceleration magnitude constraint to velocity transitions.

    Computes implied acceleration a = (requested_v - v_prev) / delta.
    If |a| exceeds ac_max, clips acceleration to +/- ac_max and computes
    actual_v = v_prev + a_clipped * delta, returning (actual_v, True).
    Otherwise returns (requested_v, False).

    Parameters:
        v_prev: Previous flight speed v_{n-1} (m/s).
        requested_v: Desired target speed v_n (m/s).
        ac_max: Maximum allowable acceleration magnitude (m/s^2).
        delta: Time slot duration in seconds.

    Returns:
        tuple[float, bool]: (actual_v, violated) where violated is True if
                            the requested change exceeded ac_max.
    """
    implied_accel = (requested_v - v_prev) / delta

    if abs(implied_accel) > ac_max:
        # Clip acceleration magnitude preserving direction
        clipped_accel = math.copysign(ac_max, implied_accel)
        actual_v = v_prev + clipped_accel * delta
        return actual_v, True

    return requested_v, False

