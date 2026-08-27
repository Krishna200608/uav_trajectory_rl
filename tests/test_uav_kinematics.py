import math
import numpy as np
from uav_trajectory_rl.uav_kinematics import apply_acceleration_constraint, step_uav_position


def test_step_position_horizontal_x():
    # lam = pi/2 (horizontal plane), rho = 0 (along +x)
    p0 = np.array([0.0, 0.0, 50.0])
    p1 = step_uav_position(p0, v=10.0, lam=0.5 * math.pi, rho=0.0, delta=1.0)
    assert np.allclose(p1, [10.0, 0.0, 50.0])


def test_step_position_horizontal_y():
    # lam = pi/2, rho = pi/2 (along +y)
    p1 = np.array([10.0, 0.0, 50.0])
    p2 = step_uav_position(p1, v=10.0, lam=0.5 * math.pi, rho=0.5 * math.pi, delta=1.0)
    assert np.allclose(p2, [10.0, 10.0, 50.0])


def test_step_position_vertical_ascent():
    # lam = 0 (along +z), rho = 0
    p2 = np.array([10.0, 10.0, 50.0])
    p3 = step_uav_position(p2, v=5.0, lam=0.0, rho=0.0, delta=2.0)
    assert np.allclose(p3, [10.0, 10.0, 60.0])


def test_step_position_vertical_descent():
    # lam = pi (along -z)
    p3 = np.array([10.0, 10.0, 60.0])
    p4 = step_uav_position(p3, v=5.0, lam=math.pi, rho=0.0, delta=1.0)
    assert np.allclose(p4, [10.0, 10.0, 55.0])


def test_step_position_hover():
    p4 = np.array([10.0, 10.0, 55.0])
    p5 = step_uav_position(p4, v=0.0, lam=0.5 * math.pi, rho=0.25 * math.pi, delta=1.0)
    assert np.allclose(p5, p4)


def test_acceleration_constraint_valid():
    v_prev = 10.0
    req_v = 13.0
    act_v, viol = apply_acceleration_constraint(v_prev, req_v, ac_max=5.0, delta=1.0)
    assert act_v == 13.0
    assert not viol


def test_acceleration_constraint_exceed_positive():
    v_prev = 10.0
    req_v_excess = 18.0
    act_v_clipped, viol_clipped = apply_acceleration_constraint(v_prev, req_v_excess, ac_max=5.0, delta=1.0)
    assert act_v_clipped == 15.0
    assert viol_clipped


def test_acceleration_constraint_exceed_negative():
    v_prev = 10.0
    req_v_brake = 0.0
    act_v_brake, viol_brake = apply_acceleration_constraint(v_prev, req_v_brake, ac_max=5.0, delta=1.0)
    assert act_v_brake == 5.0
    assert viol_brake
