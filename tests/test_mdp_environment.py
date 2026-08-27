import math
import numpy as np
from uav_trajectory_rl.config import V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def test_env_initialization_and_state_dim():
    num_users = 10
    rng = np.random.default_rng(42)
    env = UAVTrajectoryEnv(k=num_users, rng=rng)
    state = env.reset()

    expected_dim = 2 * num_users + 6  # 26
    assert state.shape == (expected_dim,)
    # UAV initial position is Q_START = [0, 0, 50]
    assert np.allclose(state[:3], [0.0, 0.0, 50.0])
    # UAV initial speed = 0.0
    assert state[-3] == 0.0
    # Remaining time = T_MAX = 200.0
    assert state[-2] == 200.0
    # Remaining distance = ||Q_END - Q_START|| = ||(600, 600, 0)|| = sqrt(720000) ~= 848.528
    assert math.isclose(state[-1], math.sqrt(600.0**2 + 600.0**2), rel_tol=1e-4)


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

    assert math.isclose(cumulative_reward, 21.8817, abs_tol=1e-2)
