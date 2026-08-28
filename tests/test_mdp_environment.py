import math
import numpy as np
import pytest

from uav_trajectory_rl.config import MAX_DISTANCE, V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def test_env_initialization_and_state_dim():
    num_users = 10
    rng = np.random.default_rng(42)
    env = UAVTrajectoryEnv(k=num_users, rng=rng)
    state = env.reset()

    expected_dim = 2 * num_users + 6  # 26
    assert state.shape == (expected_dim,)
    # UAV initial normalized position is [-1, -1, -1] corresponding to Q_START = [0, 0, 50]
    assert np.allclose(state[:3], [-1.0, -1.0, -1.0])
    # UAV initial speed = 0.0 -> v_norm = 0.0
    assert state[-3] == 0.0
    # Remaining time = T_MAX = 200.0 -> t_re_norm = 1.0
    assert state[-2] == 1.0
    # Remaining distance = ||Q_END - Q_START|| / MAX_DISTANCE ~= 848.528 / 861.684 ~= 0.98473
    expected_d_re_norm = math.sqrt(600.0**2 + 600.0**2) / MAX_DISTANCE
    assert math.isclose(state[-1], expected_d_re_norm, rel_tol=1e-4)


def test_env_20_random_steps():
    num_users = 10
    rng = np.random.default_rng(42)
    env = UAVTrajectoryEnv(k=num_users, rng=rng)
    state = env.reset()
    expected_dim = 2 * num_users + 6

    cumulative_reward = 0.0
    for i in range(1, 21):
        v_act = rng.uniform(0.0, V_MAX)
        lam_act = rng.uniform(0.0, math.pi)
        rho_act = rng.uniform(-math.pi, math.pi)

        next_state, reward, done, info = env.step((v_act, lam_act, rho_act))
        cumulative_reward += reward

        assert next_state.shape == (expected_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "r1_throughput" in info
        assert "r2_energy" in info
        assert "r3_terminal" in info
        assert "r4_proximity" in info
        assert "r5_accel" in info
        assert "r6_height" in info
        if done:
            break

    # Cumulative reward uses raw physical attributes, unaffected by state normalization
    assert math.isclose(cumulative_reward, 21.8817, abs_tol=1e-2)


def test_state_normalization_bounds():
    """
    Confirm every component of freshly-reset and post-step state vectors
    lies within [-1.5, 1.5] across multiple random seeds.
    """
    for seed in [1, 42, 99, 123, 777]:
        rng = np.random.default_rng(seed)
        env = UAVTrajectoryEnv(k=10, rng=rng)
        state = env.reset()

        assert np.all(state >= -1.5) and np.all(state <= 1.5)

        for _ in range(30):
            v_act = rng.uniform(0.0, V_MAX)
            lam_act = rng.uniform(0.0, math.pi)
            rho_act = rng.uniform(-math.pi, math.pi)
            next_state, _, done, _ = env.step((v_act, lam_act, rho_act))

            assert np.all(next_state >= -1.5) and np.all(next_state <= 1.5)
            if done:
                break


def test_charge_energy_on_cancelled_move_toggle():
    """
    Verify that when an action violates spatial constraints and gets cancelled:
    - With charge_energy_on_cancelled_move=True: energy r2 is charged based on commanded v.
    - With charge_energy_on_cancelled_move=False: energy r2 uses v=0.0 (hovering energy only).
    """
    # Action pushing horizontally west past x=0 boundary: lam = pi/2, rho = pi, v = 20
    action_into_wall = (V_MAX, math.pi / 2.0, math.pi)

    env_charge = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0), charge_energy_on_cancelled_move=True)
    env_charge.reset()
    _, _, _, info_charge = env_charge.step(action_into_wall)

    env_no_charge = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0), charge_energy_on_cancelled_move=False)
    env_no_charge.reset()
    _, _, _, info_no_charge = env_no_charge.step(action_into_wall)

    # Both environments cancel movement and remain at Q_START
    assert np.allclose(env_charge.uav_pos, env_charge.q_start)
    assert np.allclose(env_no_charge.uav_pos, env_no_charge.q_start)

    # env_no_charge charged energy based on v=0.0 (hovering)
    # env_charge charged energy based on actual_v (acceleration-capped commanded speed)
    expected_no_charge_r2 = env_no_charge._reward_energy(0.0, math.pi / 2.0)
    expected_charge_r2 = env_charge._reward_energy(env_charge.uav_speed, math.pi / 2.0)
    assert math.isclose(info_no_charge["r2_energy"], expected_no_charge_r2)
    assert math.isclose(info_charge["r2_energy"], expected_charge_r2)
    assert not math.isclose(info_no_charge["r2_energy"], info_charge["r2_energy"])
