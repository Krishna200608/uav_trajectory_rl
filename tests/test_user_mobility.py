import numpy as np
from uav_trajectory_rl.user_mobility import UserSwarm


def test_user_swarm_initialization():
    num_users = 10
    area = (0.0, 600.0, 0.0, 600.0)
    rng = np.random.default_rng(42)

    swarm = UserSwarm(k=num_users, area_bounds=area, rng=rng)
    positions = swarm.get_positions()
    velocities = swarm.get_velocities()
    directions = swarm.get_directions()

    assert positions.shape == (num_users, 2)
    assert velocities.shape == (num_users,)
    assert directions.shape == (num_users,)

    # Check bounds
    assert np.all(positions[:, 0] >= area[0]) and np.all(positions[:, 0] <= area[1])
    assert np.all(positions[:, 1] >= area[2]) and np.all(positions[:, 1] <= area[3])
    assert np.all(velocities >= 0.0)
    assert np.all(directions >= -np.pi) and np.all(directions <= np.pi)


def test_user_swarm_steps_and_bounds():
    num_users = 10
    area = (0.0, 600.0, 0.0, 600.0)
    rng = np.random.default_rng(42)

    swarm = UserSwarm(k=num_users, area_bounds=area, rng=rng)

    for _ in range(5):
        pos = swarm.step()
        assert pos.shape == (num_users, 2)
        assert np.all(pos[:, 0] >= area[0]) and np.all(pos[:, 0] <= area[1])
        assert np.all(pos[:, 1] >= area[2]) and np.all(pos[:, 1] <= area[3])
