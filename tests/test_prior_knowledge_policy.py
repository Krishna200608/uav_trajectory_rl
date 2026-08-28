import math
import numpy as np
import pytest

from uav_trajectory_rl.config import (
    ACTION_CLIP_C,
    LAMBDA_PK,
    RHO_PK,
    R_RAND,
    V_MAX,
)
from uav_trajectory_rl.prior_knowledge_policy import (
    generate_prior_knowledge_action,
    normalize_action,
    select_action,
    unnormalize_action,
)


def test_generate_prior_knowledge_action():
    rng = np.random.default_rng(42)
    for _ in range(200):
        v, lam, rho = generate_prior_knowledge_action(rng)
        assert 0.0 <= v <= V_MAX
        assert lam == LAMBDA_PK
        assert 0.0 <= rho <= RHO_PK  # strictly non-negative per eq. 30


def test_unnormalize_action():
    c = 1.0  # matches Table III ACTION_CLIP_C

    # Center
    v, lam, rho = unnormalize_action(np.array([0.0, 0.0, 0.0]), c=c)
    assert math.isclose(v, V_MAX / 2.0)
    assert math.isclose(lam, math.pi / 2.0)
    assert math.isclose(rho, 0.0)

    # Lower bound [-1, -1, -1]
    v_min, lam_min, rho_min = unnormalize_action(np.array([-1.0, -1.0, -1.0]), c=c)
    assert math.isclose(v_min, 0.0)
    assert math.isclose(lam_min, 0.0)
    assert math.isclose(rho_min, -math.pi)

    # Upper bound [1, 1, 1]
    v_max, lam_max, rho_max = unnormalize_action(np.array([1.0, 1.0, 1.0]), c=c)
    assert math.isclose(v_max, V_MAX)
    assert math.isclose(lam_max, math.pi)
    assert math.isclose(rho_max, math.pi)


def test_select_action_prior_knowledge_branch():
    rng = np.random.default_rng(123)
    dummy_state = np.zeros(26)

    def exploding_actor_fn(state: np.ndarray) -> np.ndarray:
        raise AssertionError("actor_fn should NOT be called in prior-knowledge branch!")

    # Test both 0 and R_RAND (boundary)
    for buf_size in [0, 500, R_RAND]:
        v, lam, rho = select_action(
            state=dummy_state,
            replay_buffer_size=buf_size,
            actor_fn=exploding_actor_fn,
            rng=rng,
        )
        assert 0.0 <= v <= V_MAX
        assert lam == LAMBDA_PK
        assert 0.0 <= rho <= RHO_PK


def test_select_action_network_branch():
    rng = np.random.default_rng(456)
    dummy_state = np.zeros(26)

    def mock_actor_fn(state: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    # Stress test with huge exploration noise sigma3=10.0 to force heavy clipping
    for _ in range(200):
        v, lam, rho = select_action(
            state=dummy_state,
            replay_buffer_size=R_RAND + 1,
            actor_fn=mock_actor_fn,
            rng=rng,
            sigma3=10.0,
            c=ACTION_CLIP_C,
        )
        assert 0.0 <= v <= V_MAX
        assert 0.0 <= lam <= math.pi
        assert -math.pi <= rho <= math.pi


def test_action_normalization_round_trip():
    """
    Confirm exact mathematical round-trip between normalize_action and unnormalize_action.
    1. raw -> unnormalize -> normalize -> raw
    2. phys -> normalize -> unnormalize -> phys
    """
    rng = np.random.default_rng(789)
    c = ACTION_CLIP_C

    # 1. Random raw actions in [-c, c]^3
    for _ in range(500):
        raw = rng.uniform(-c, c, size=3)
        phys = unnormalize_action(raw, c=c)
        recovered_raw = normalize_action(phys, c=c)
        assert np.allclose(recovered_raw, raw, atol=1e-9)

    # 2. Random physical actions in valid physical bounds
    for _ in range(500):
        v = rng.uniform(0.0, V_MAX)
        lam = rng.uniform(0.0, math.pi)
        rho = rng.uniform(-math.pi, math.pi)
        phys = (v, lam, rho)
        norm = normalize_action(phys, c=c)
        recovered_phys = unnormalize_action(norm, c=c)
        assert np.allclose(recovered_phys, phys, atol=1e-9)
