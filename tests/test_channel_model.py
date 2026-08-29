import math
import numpy as np
from uav_trajectory_rl.channel_model import (
    average_path_loss,
    los_probability,
    nlos_probability,
    path_loss_los,
    path_loss_nlos,
    total_transmission_rate,
    transmission_rate,
)


def test_los_nlos_probabilities():
    z_diff = 100.0
    h_dist = math.sqrt(50.0**2 + 20.0**2)
    p_los = los_probability(z_diff, h_dist)
    p_nlos = nlos_probability(p_los)

    assert 0.0 < p_los <= 1.0
    assert math.isclose(p_los + p_nlos, 1.0)
    assert math.isclose(p_los, 0.9977, abs_tol=1e-3)


def test_path_losses():
    z_diff = 100.0
    h_dist = math.sqrt(50.0**2 + 20.0**2)
    r_3d = math.sqrt(h_dist**2 + z_diff**2)

    pl_los = path_loss_los(r_3d)
    pl_nlos = path_loss_nlos(r_3d)
    pl_avg = average_path_loss(z_diff, h_dist)

    assert pl_nlos > pl_los
    assert math.isclose(pl_los, 82.15, abs_tol=0.1)
    assert math.isclose(pl_nlos, 101.15, abs_tol=0.1)
    assert math.isclose(pl_avg, 82.20, abs_tol=0.1)


def test_transmission_rate_single_user():
    uav_pos = np.array([300.0, 300.0, 100.0])
    user_pos = np.array([250.0, 280.0, 0.0])

    rate = transmission_rate(uav_pos, user_pos, num_users_k=1)
    assert rate > 0.0
    # Paper literal eq. (13) with log2(SNR)
    assert math.isclose(rate, 191303475.88, rel_tol=1e-3)


def test_total_transmission_rate_group():
    uav_pos = np.array([300.0, 300.0, 100.0])
    users_group = np.array([
        [250.0, 280.0, 0.0],
        [320.0, 310.0, 0.0],
        [400.0, 200.0, 0.0],
        [150.0, 450.0, 0.0],
    ])
    sum_rate = total_transmission_rate(uav_pos, users_group)
    assert sum_rate > 0.0
    assert math.isclose(sum_rate, 198223227.00, rel_tol=1e-3)
