"""
Rotary-Wing UAV Propulsion Power and Energy Consumption Model.

This module implements the aerodynamic propulsion energy model for rotary-wing
UAVs based on the Cai/Zeng formulation referenced in eq. (15)-(16) of the IEEE TNSE
paper:
    "3-D Trajectory Design Based on Deep Reinforcement Learning for
     UAV-Assisted Communication Networks"

Key Physical Components:
    1. Blade profile power (overcoming profile drag of rotating blades)
    2. Induced power (overcoming induced aerodynamic drag and producing lift)
    3. Climb / descent power (potential energy rate depending on pitch angle)
    4. Parasitic fuselage drag power (air resistance of the moving aircraft frame)
"""

import math
from typing import Optional

try:
    from uav_trajectory_rl.config import (
        DELTA,
        UAV_A,
        UAV_C,
        UAV_D0,
        UAV_P0,
        UAV_RHO,
        UAV_S,
        UAV_SFP,
        UAV_UTIP,
        UAV_W,
    )
except ImportError:
    from config import (
        DELTA,
        UAV_A,
        UAV_C,
        UAV_D0,
        UAV_P0,
        UAV_RHO,
        UAV_S,
        UAV_SFP,
        UAV_UTIP,
        UAV_W,
    )


def induced_velocity_hover(
    uav_w: float = UAV_W,
    uav_rho: float = UAV_RHO,
    uav_a: float = UAV_A,
) -> float:
    """
    Compute mean rotor-induced velocity in hovering state (V0) via momentum theory.

    Formula:
        V0 = sqrt(W / (2 * rho * A))

    Parameters:
        uav_w: UAV weight in Newtons (W).
        uav_rho: Air density in kg/m^3 (rho).
        uav_a: Rotor disc area in m^2 (A).

    Returns:
        float: Hover induced velocity V0 (m/s).
    """
    denominator = 2.0 * uav_rho * uav_a
    if denominator <= 0.0:
        raise ValueError(f"Invalid density/rotor area: rho={uav_rho}, A={uav_a}")
    return float(math.sqrt(uav_w / denominator))


def propulsion_power(
    v_n: float,
    lam_n: float,
    uav_w: float = UAV_W,
    uav_rho: float = UAV_RHO,
    uav_a: float = UAV_A,
    uav_p0: float = UAV_P0,
    uav_utip: float = UAV_UTIP,
    uav_c: float = UAV_C,
    uav_sfp: float = UAV_SFP,
    uav_s: float = UAV_S,
    uav_d0: float = UAV_D0,
    v0: Optional[float] = None,
) -> float:
    """
    Compute rotary-wing UAV instantaneous propulsion power in Watts (eq. 15).

    Formulation:
        - Pitch angle: tau_c = pi/2 - lam_n
        - ASSUMPTION: Longitudinal flight speed V_y is modeled as the horizontal
          velocity component: V_y = v_n * sin(lam_n).
        - Longitudinal air resistance: F_y = 0.5 * C * rho * S_FP * V_y^2
        - Induced power correction factor (m_tilde): Computed dynamically via
          m_tilde = (W - F_y) / (W * cos(tau_c)) as given in the paper text.
        - Power breakdown:
          P = P_blade + P_induced + P_climb + P_parasite

    Parameters:
        v_n: Flight speed magnitude in m/s.
        lam_n: Polar angle in radians [0, pi] (from positive z-axis).
        uav_w: UAV weight (N).
        uav_rho: Air density (kg/m^3).
        uav_a: Rotor disc area (m^2).
        uav_p0: Blade profile power in hover (W).
        uav_utip: Rotor tip speed (m/s).
        uav_c: Air resistance coefficient.
        uav_sfp: Fuselage equivalent flat-plate area (m^2).
        uav_s: Rotor solidity.
        uav_d0: Fuselage drag ratio.
        v0: Hover induced velocity (m/s). If None, calculated via momentum theory.

    Returns:
        float: Instantaneous propulsion power P in Watts (W).
    """
    if v0 is None:
        v0 = induced_velocity_hover(uav_w=uav_w, uav_rho=uav_rho, uav_a=uav_a)

    # 1. Pitch angle tau_c relative to horizontal xy-plane
    tau_c = 0.5 * math.pi - lam_n
    cos_tau = math.cos(tau_c)
    sin_tau = math.sin(tau_c)

    # Safe cos_tau to prevent zero-division in pure vertical flight
    safe_cos_tau = cos_tau if abs(cos_tau) > 1e-7 else math.copysign(1e-7, cos_tau if cos_tau != 0 else 1.0)
    tan_tau = sin_tau / safe_cos_tau

    # 2. Longitudinal velocity component V_y and longitudinal air resistance F_y
    # ASSUMPTION: V_y corresponds to the horizontal speed component v_n * sin(lam_n)
    v_y = v_n * math.sin(lam_n)
    f_y = 0.5 * uav_c * uav_rho * uav_sfp * (v_y**2)

    # 3. Induced-power correction factor (m_tilde)
    # Computed from the paper text formula: m_tilde = (W - F_y) / (W * cos(tau_c))
    m_tilde = (uav_w - f_y) / (uav_w * safe_cos_tau)

    # --------------------------------------------------------------------------
    # Power Term 1: Blade Profile Power
    # --------------------------------------------------------------------------
    p_blade = uav_p0 * (1.0 + (3.0 * (v_n**2)) / (uav_utip**2))

    # --------------------------------------------------------------------------
    # Power Term 2: Induced Power
    # --------------------------------------------------------------------------
    effective_weight = max(0.0, uav_w - (f_y / safe_cos_tau))
    weight_factor = (effective_weight**1.5) / math.sqrt(2.0 * uav_rho * uav_a)

    term_v0 = (v_n**2) / (2.0 * (v0**2))
    inner_sqrt_arg = max(0.0, (m_tilde**2) + ((v_n**4) / (4.0 * (v0**4))))
    induced_bracket = max(0.0, math.sqrt(inner_sqrt_arg) - term_v0)
    induced_speed_factor = math.sqrt(induced_bracket)

    p_induced = (1.0 + m_tilde) * weight_factor * induced_speed_factor

    # --------------------------------------------------------------------------
    # Power Term 3: Climb / Descent Power
    # --------------------------------------------------------------------------
    p_climb = (uav_w - f_y) * v_n * tan_tau

    # --------------------------------------------------------------------------
    # Power Term 4: Parasite Fuselage Drag Power
    # --------------------------------------------------------------------------
    p_parasite = 0.5 * uav_rho * uav_s * uav_a * (v_n**3) * uav_d0 * (cos_tau**2)

    total_power = p_blade + p_induced + p_climb + p_parasite
    return float(total_power)


def energy_consumption(
    v_n: float,
    lam_n: float,
    delta: float = DELTA,
    **kwargs,
) -> float:
    """
    Calculate propulsion energy consumption during one time slot in Joules (eq. 16).

    Formula:
        E_n = P(tau_c, F_y, v_n) * delta

    Parameters:
        v_n: Flight speed (m/s).
        lam_n: Polar angle (radians).
        delta: Time slot duration in seconds (default from config: DELTA).
        **kwargs: Additional optional overrides for aircraft parameters.

    Returns:
        float: Energy consumed in Joules (J).
    """
    power_w = propulsion_power(v_n=v_n, lam_n=lam_n, **kwargs)
    return float(power_w * delta)


if __name__ == "__main__":
    print("=" * 60)
    print("UAV Propulsion Power & Energy Model Demo")
    print("=" * 60)

    test_v = 10.0            # m/s
    test_lam = 0.25 * math.pi # pi / 4 rad

    v0_derived = induced_velocity_hover()
    power_watts = propulsion_power(test_v, test_lam)
    energy_joules = energy_consumption(test_v, test_lam)

    print(f"Rotor Induced Velocity in Hover (V0): {v0_derived:.4f} m/s")
    print(f"Flight Speed (v_n):                   {test_v:.2f} m/s")
    print(f"Polar Angle (lambda_n):               {test_lam:.4f} rad ({test_lam * 180 / math.pi:.1f} deg)")
    print(f"Propulsion Power (P_n):               {power_watts:.2f} Watts")
    print(f"Energy for delta={DELTA:.1f}s (E_n):          {energy_joules:.2f} Joules")

    # Sanity checks
    assert v0_derived > 0.0, "Hover velocity must be positive"
    assert power_watts > 0.0, "Propulsion power must be positive"
    assert energy_joules > 0.0, "Energy consumption must be positive"

    # Hover comparison (v = 0, lam = pi/2)
    hover_power = propulsion_power(0.0, 0.5 * math.pi)
    print(f"\nHover Power (v=0, lam=pi/2):          {hover_power:.2f} Watts")

    print("\nEnergy Model verification PASSED successfully!")
