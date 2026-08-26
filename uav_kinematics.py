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

try:
    from uav_trajectory_rl.config import V_MAX, AC_MAX, DELTA
except ImportError:
    from config import V_MAX, AC_MAX, DELTA


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


if __name__ == "__main__":
    print("=" * 60)
    print("UAV Kinematics Module Sanity Checks")
    print("=" * 60)

    # 1. Test Step Position: Horizontal motion along X-axis
    # lam = pi/2 (horizontal plane), rho = 0 (along +x)
    p0 = np.array([0.0, 0.0, 50.0])
    v = 10.0
    lam = 0.5 * math.pi
    rho = 0.0
    p1 = step_uav_position(p0, v=v, lam=lam, rho=rho, delta=1.0)
    print(f"Initial: {p0} -> Action (v={v}, lam=pi/2, rho=0) -> Next: {p1}")
    assert np.allclose(p1, [10.0, 0.0, 50.0]), f"Expected [10, 0, 50], got {p1}"

    # 2. Test Step Position: Horizontal motion along Y-axis
    # lam = pi/2, rho = pi/2 (along +y)
    p2 = step_uav_position(p1, v=10.0, lam=0.5 * math.pi, rho=0.5 * math.pi, delta=1.0)
    print(f"Initial: {p1} -> Action (v=10, lam=pi/2, rho=pi/2) -> Next: {p2}")
    assert np.allclose(p2, [10.0, 10.0, 50.0]), f"Expected [10, 10, 50], got {p2}"

    # 3. Test Step Position: Pure vertical ascent along Z-axis
    # lam = 0 (along +z), rho = 0
    p3 = step_uav_position(p2, v=5.0, lam=0.0, rho=0.0, delta=2.0)
    print(f"Initial: {p2} -> Action (v=5, lam=0, rho=0, delta=2) -> Next: {p3}")
    assert np.allclose(p3, [10.0, 10.0, 60.0]), f"Expected [10, 10, 60], got {p3}"

    # 4. Test Step Position: Pure vertical descent along Z-axis
    # lam = pi (along -z)
    p4 = step_uav_position(p3, v=5.0, lam=math.pi, rho=0.0, delta=1.0)
    print(f"Initial: {p3} -> Action (v=5, lam=pi, rho=0, delta=1) -> Next: {p4}")
    assert np.allclose(p4, [10.0, 10.0, 55.0]), f"Expected [10, 10, 55], got {p4}"

    # 5. Test Step Position: Hovering (v = 0)
    p5 = step_uav_position(p4, v=0.0, lam=0.5 * math.pi, rho=0.25 * math.pi, delta=1.0)
    print(f"Initial: {p4} -> Hover (v=0) -> Next: {p5}")
    assert np.allclose(p5, p4), f"Expected unchanged position on hover, got {p5}"

    # 6. Test Acceleration Constraints
    # Case A: Valid acceleration within AC_MAX (5.0 m/s^2)
    v_prev = 10.0
    req_v = 13.0  # accel = +3.0 m/s^2 <= 5.0
    act_v, viol = apply_acceleration_constraint(v_prev, req_v, ac_max=5.0, delta=1.0)
    print(f"\nAccel Check (Valid +): v_prev={v_prev}, req={req_v} -> act={act_v}, viol={viol}")
    assert act_v == 13.0 and not viol, "Failed valid acceleration check"

    # Case B: Exceeding positive acceleration limit (req change = +8.0 m/s^2 > 5.0)
    req_v_excess = 18.0
    act_v_clipped, viol_clipped = apply_acceleration_constraint(v_prev, req_v_excess, ac_max=5.0, delta=1.0)
    print(f"Accel Check (Exceed +): v_prev={v_prev}, req={req_v_excess} -> act={act_v_clipped}, viol={viol_clipped}")
    assert act_v_clipped == 15.0 and viol_clipped, f"Expected 15.0 with viol=True, got {act_v_clipped}, {viol_clipped}"

    # Case C: Exceeding deceleration limit (req change = -10.0 m/s^2 < -5.0)
    req_v_brake = 0.0
    act_v_brake, viol_brake = apply_acceleration_constraint(v_prev, req_v_brake, ac_max=5.0, delta=1.0)
    print(f"Accel Check (Exceed -): v_prev={v_prev}, req={req_v_brake} -> act={act_v_brake}, viol={viol_brake}")
    assert act_v_brake == 5.0 and viol_brake, f"Expected 5.0 with viol=True, got {act_v_brake}, {viol_brake}"

    print("\nAll UAV Kinematics tests PASSED successfully!")
