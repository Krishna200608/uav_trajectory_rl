"""
Unit tests for Sweep vs. User Mobility Speed (Module M14e).
"""

from pathlib import Path

import numpy as np
import pytest

from uav_trajectory_rl.evaluation.figures_9 import (
    generate_fig9_speed_sweep,
    speed_to_range,
)


def test_speed_to_range_mapping():
    """
    Verify speed_to_range properly maps scalar mobility speeds to +/- 1.0 m/s intervals
    and correctly enforces the 0.1 m/s floor.
    """
    # Standard nominal case
    lo, hi = speed_to_range(2.0)
    assert np.isclose(lo, 1.0)
    assert np.isclose(hi, 3.0)

    # Low-end case exercising the max(0.1, ...) floor
    lo_floor, hi_floor = speed_to_range(0.5)
    assert np.isclose(lo_floor, 0.1)
    assert np.isclose(hi_floor, 1.5)

    # Zero speed boundary
    lo_zero, hi_zero = speed_to_range(0.0)
    assert np.isclose(lo_zero, 0.1)
    assert np.isclose(hi_zero, 1.0)

    # High-end speed case
    lo_hi, hi_hi = speed_to_range(12.0)
    assert np.isclose(lo_hi, 11.0)
    assert np.isclose(hi_hi, 13.0)


def test_generate_fig9_creates_file(tmp_path: Path):
    """
    Verify generate_fig9_speed_sweep creates a non-empty PNG image.
    Uses small parameter range (speed_values=(2, 6), 2 seeds) for test execution speed.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    fig_path = generate_fig9_speed_sweep(
        speed_values=(2, 6),
        sweep_seeds=range(2),
        k=10,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )

    assert fig_path.exists()
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000
    assert fig_path.name == "fig9_speed_sweep.png"
