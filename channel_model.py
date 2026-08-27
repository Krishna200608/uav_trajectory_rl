"""
Air-to-Ground (A2G) Channel Model and Achievable Rate Calculations.

This module implements the probabilistic Line-of-Sight (LoS) / Non-Line-of-Sight (NLoS)
path loss and transmission rate models for UAV-to-ground-user communications,
corresponding to equations (8)-(14) of the IEEE TNSE paper:
    "3-D Trajectory Design Based on Deep Reinforcement Learning for
     UAV-Assisted Communication Networks"

Unit Consistency Note:
    - Transmit power (p_k): Converted from P_K_DBM to linear milliwatts (mW).
    - Thermal noise (N0): Converted from N0_DBM_HZ to linear milliwatts per Hertz (mW/Hz).
    - Bandwidth (B_u / K): Hertz (Hz).
    - Noise power: N0 * (B_u / K) in milliwatts (mW).
    - Received SNR: (p_k / beta) / (N0 * B_u / K) is dimensionless (mW / mW).
    - Achievable rate: Bits per second (bps), computed as (B_u / K) * log2(SNR) per literal eq. (13).
"""

import math
from typing import Sequence, Union
import numpy as np

try:
    from uav_trajectory_rl.config import (
        B1,
        B2,
        B_U_HZ,
        ETA_LOS_DB,
        ETA_NLOS_DB,
        FC_HZ,
        N0_DBM_HZ,
        P_K_DBM,
        VC,
    )
except ImportError:
    from config import (
        B1,
        B2,
        B_U_HZ,
        ETA_LOS_DB,
        ETA_NLOS_DB,
        FC_HZ,
        N0_DBM_HZ,
        P_K_DBM,
        VC,
    )


def los_probability(
    z_uav: float,
    horizontal_dist_to_user: float,
    b1: float = B1,
    b2: float = B2,
) -> float:
    """
    Calculate the Line-of-Sight (LoS) probability between UAV and ground user (eq. 8-9).

    Parameters:
        z_uav: Relative altitude difference between UAV and user (m).
        horizontal_dist_to_user: 2D horizontal Euclidean distance to user (m).
        b1: Environmental parameter.
        b2: Environmental parameter.

    Returns:
        float: Probability of Line-of-Sight connection P_LoS in [0, 1].
    """
    r = math.sqrt(z_uav**2 + horizontal_dist_to_user**2)
    if r == 0.0:
        return 1.0

    # Elevation angle in degrees: theta = (180 / pi) * arcsin(z / r)
    sin_elevation = min(1.0, max(-1.0, z_uav / r))
    elevation_deg = (180.0 / math.pi) * math.asin(sin_elevation)

    # Sigmoidal LoS probability formula (eq. 8)
    exp_arg = -b2 * (elevation_deg - b1)
    # Clip exp_arg to prevent overflow
    exp_arg = max(-100.0, min(100.0, exp_arg))
    p_los = 1.0 / (1.0 + b1 * math.exp(exp_arg))

    return float(p_los)


def nlos_probability(p_los: float) -> float:
    """
    Calculate Non-Line-of-Sight (NLoS) probability (eq. 10).

    Parameters:
        p_los: Line-of-Sight probability in [0, 1].

    Returns:
        float: Probability of Non-Line-of-Sight connection P_NLoS = 1 - P_LoS.
    """
    return 1.0 - p_los


def free_space_path_loss(
    r: float,
    fc_hz: float = FC_HZ,
    vc: float = VC,
) -> float:
    """
    Calculate Free Space Path Loss (FSPL) in dB (eq. 11).

    LF(r) = 20 * log10(r) + 20 * log10(fc) + 20 * log10(4 * pi / vc)

    Parameters:
        r: 3D propagation distance in meters.
        fc_hz: Carrier frequency in Hz.
        vc: Speed of light in m/s.

    Returns:
        float: Free space path loss in dB.
    """
    if r <= 0.0:
        r = 1e-6  # Prevent log(0) singularity

    constant_term = 20.0 * math.log10(4.0 * math.pi / vc)
    lf = 20.0 * math.log10(r) + 20.0 * math.log10(fc_hz) + constant_term
    return float(lf)


def path_loss_los(
    r: float,
    fc_hz: float = FC_HZ,
    vc: float = VC,
    eta_los_db: float = ETA_LOS_DB,
) -> float:
    """
    Calculate Line-of-Sight (LoS) path loss in dB.

    PL_LoS = LF(r) + eta_LoS
    """
    return free_space_path_loss(r, fc_hz=fc_hz, vc=vc) + eta_los_db


def path_loss_nlos(
    r: float,
    fc_hz: float = FC_HZ,
    vc: float = VC,
    eta_nlos_db: float = ETA_NLOS_DB,
) -> float:
    """
    Calculate Non-Line-of-Sight (NLoS) path loss in dB.

    PL_NLoS = LF(r) + eta_NLoS
    """
    return free_space_path_loss(r, fc_hz=fc_hz, vc=vc) + eta_nlos_db


def average_path_loss(
    z_uav: float,
    horizontal_dist: float,
    fc_hz: float = FC_HZ,
    vc: float = VC,
    b1: float = B1,
    b2: float = B2,
    eta_los_db: float = ETA_LOS_DB,
    eta_nlos_db: float = ETA_NLOS_DB,
) -> float:
    """
    Calculate average path loss in dB as a linear combination of LoS/NLoS dB losses (eq. 12).

    PL_avg = P_LoS * PL_LoS + P_NLoS * PL_NLoS

    Parameters:
        z_uav: Relative altitude difference (m).
        horizontal_dist: Horizontal 2D distance (m).
        fc_hz: Carrier frequency in Hz.
        vc: Speed of light in m/s.

    Returns:
        float: Average path loss in dB.
    """
    r = math.sqrt(z_uav**2 + horizontal_dist**2)
    p_los = los_probability(z_uav, horizontal_dist, b1=b1, b2=b2)
    p_nlos = nlos_probability(p_los)

    pl_los = path_loss_los(r, fc_hz=fc_hz, vc=vc, eta_los_db=eta_los_db)
    pl_nlos = path_loss_nlos(r, fc_hz=fc_hz, vc=vc, eta_nlos_db=eta_nlos_db)

    # Linear combination of dB losses as per source paper formulation eq. (12)
    pl_avg_db = p_los * pl_los + p_nlos * pl_nlos
    return float(pl_avg_db)


def transmission_rate(
    uav_pos: Union[Sequence[float], np.ndarray],
    user_pos: Union[Sequence[float], np.ndarray],
    num_users_k: int,
    p_k_dbm: float = P_K_DBM,
    bu_hz: float = B_U_HZ,
    n0_dbm_hz: float = N0_DBM_HZ,
    fc_hz: float = FC_HZ,
    vc: float = VC,
) -> float:
    """
    Compute achievable transmission rate for user k in bits/s (literal eq. 13).

    Formula:
        R_n^k = (B_u / K) * log2(p_k / (beta_n^k * N0 * (B_u / K)))
        where beta_n^k = 10^(PL_n^k / 10) (average path loss in linear scale).
        Note: Per literal eq. (13) of the paper, this is log2(SNR) without '+ 1'.
        When SNR < 1, rate can be negative.

    Parameters:
        uav_pos: 3D UAV position [x, y, z] in meters.
        user_pos: 2D or 3D user position [x, y] or [x, y, z] in meters.
        num_users_k: Total number of active ground users K sharing the bandwidth.
        p_k_dbm: User transmit power in dBm.
        bu_hz: Total UAV bandwidth in Hz.
        n0_dbm_hz: Noise power spectral density in dBm/Hz.
        fc_hz: Carrier frequency in Hz.
        vc: Speed of light in m/s.

    Returns:
        float: Achievable transmission rate R_n^k in bits per second (bps).
    """
    if num_users_k <= 0:
        return 0.0

    # Calculate spatial geometry
    dx = uav_pos[0] - user_pos[0]
    dy = uav_pos[1] - user_pos[1]
    horizontal_dist = math.sqrt(dx**2 + dy**2)

    user_z = user_pos[2] if len(user_pos) > 2 else 0.0
    relative_z = uav_pos[2] - user_z

    # Compute average path loss in dB and convert to linear scale beta
    pl_db = average_path_loss(relative_z, horizontal_dist, fc_hz=fc_hz, vc=vc)
    beta_linear = 10.0 ** (pl_db / 10.0)

    # Unit conversions: dBm -> linear milliwatts (mW)
    p_k_mw = 10.0 ** (p_k_dbm / 10.0)
    n0_mw_per_hz = 10.0 ** (n0_dbm_hz / 10.0)

    # Bandwidth allocated per user (FDMA)
    bandwidth_per_user = bu_hz / float(num_users_k)

    # Total effective noise power in mW
    noise_power_mw = beta_linear * n0_mw_per_hz * bandwidth_per_user

    # Signal-to-Noise Ratio (dimensionless: mW / mW)
    snr = p_k_mw / noise_power_mw

    # Transmission rate per paper literal eq. (13): log2(SNR) without '+ 1'
    rate_bps = bandwidth_per_user * math.log2(snr)
    return float(rate_bps)


def total_transmission_rate(
    uav_pos: Union[Sequence[float], np.ndarray],
    user_positions: Union[Sequence[Sequence[float]], np.ndarray],
    p_k_dbm: float = P_K_DBM,
    bu_hz: float = B_U_HZ,
    n0_dbm_hz: float = N0_DBM_HZ,
    fc_hz: float = FC_HZ,
    vc: float = VC,
) -> float:
    """
    Calculate sum transmission rate over all ground users in bits/s (eq. 14).

    Parameters:
        uav_pos: 3D UAV position [x, y, z] in meters.
        user_positions: Sequence or (K, 2)/(K, 3) array of user positions.

    Returns:
        float: Total system throughput in bits per second (bps).
    """
    k = len(user_positions)
    if k == 0:
        return 0.0

    total_rate = 0.0
    for pos in user_positions:
        total_rate += transmission_rate(
            uav_pos=uav_pos,
            user_pos=pos,
            num_users_k=k,
            p_k_dbm=p_k_dbm,
            bu_hz=bu_hz,
            n0_dbm_hz=n0_dbm_hz,
            fc_hz=fc_hz,
            vc=vc,
        )

    return float(total_rate)


if __name__ == "__main__":
    print("=" * 60)
    print("Air-to-Ground (A2G) Channel Model Sanity Check")
    print("=" * 60)

    uav_test_pos = np.array([300.0, 300.0, 100.0])
    user_test_pos = np.array([250.0, 280.0, 0.0])

    dx = uav_test_pos[0] - user_test_pos[0]
    dy = uav_test_pos[1] - user_test_pos[1]
    h_dist = math.sqrt(dx**2 + dy**2)
    z_diff = uav_test_pos[2] - user_test_pos[2]
    r_3d = math.sqrt(h_dist**2 + z_diff**2)

    p_los = los_probability(z_diff, h_dist)
    p_nlos = nlos_probability(p_los)
    pl_los = path_loss_los(r_3d)
    pl_nlos = path_loss_nlos(r_3d)
    pl_avg = average_path_loss(z_diff, h_dist)
    single_user_rate = transmission_rate(uav_test_pos, user_test_pos, num_users_k=1)

    print(f"UAV Position:        {uav_test_pos} m")
    print(f"User Position:       {user_test_pos} m")
    print(f"Horizontal Distance: {h_dist:.2f} m")
    print(f"3D Propagation Dist: {r_3d:.2f} m")
    print(f"LoS Probability:     {p_los:.4f} ({p_los * 100:.2f}%)")
    print(f"NLoS Probability:    {p_nlos:.4f} ({p_nlos * 100:.2f}%)")
    print(f"LoS Path Loss:       {pl_los:.2f} dB")
    print(f"NLoS Path Loss:      {pl_nlos:.2f} dB")
    print(f"Average Path Loss:   {pl_avg:.2f} dB")
    print(f"Transmission Rate:   {single_user_rate:,.2f} bps ({single_user_rate / 1e6:.2f} Mbps)")

    # Multiple users sum rate check
    users_group = np.array([
        [250.0, 280.0, 0.0],
        [320.0, 310.0, 0.0],
        [400.0, 200.0, 0.0],
        [150.0, 450.0, 0.0],
    ])
    sum_rate = total_transmission_rate(uav_test_pos, users_group)
    print(f"\nTotal Rate for {len(users_group)} users: {sum_rate:,.2f} bps ({sum_rate / 1e6:.2f} Mbps)")

    assert p_los > 0.0 and p_los <= 1.0, "LoS probability out of bounds"
    assert single_user_rate > 0.0, "Transmission rate should be positive"
    print("\nChannel Model verification PASSED successfully!")
