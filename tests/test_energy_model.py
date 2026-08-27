import math
from uav_trajectory_rl.energy_model import (
    energy_consumption,
    induced_velocity_hover,
    propulsion_power,
)


def test_induced_velocity_hover():
    v0 = induced_velocity_hover()
    assert v0 > 0.0
    assert math.isclose(v0, 4.0203, abs_tol=1e-3)


def test_propulsion_power_and_energy():
    test_v = 10.0
    test_lam = 0.25 * math.pi
    power = propulsion_power(test_v, test_lam)
    energy = energy_consumption(test_v, test_lam, delta=1.0)

    assert power > 0.0
    assert energy > 0.0
    assert math.isclose(power, 396.34, abs_tol=0.1)
    assert math.isclose(energy, 396.34, abs_tol=0.1)


def test_hover_power():
    hover_p = propulsion_power(0.0, 0.5 * math.pi)
    assert hover_p > 0.0
    assert math.isclose(hover_p, 241.00, abs_tol=0.1)
