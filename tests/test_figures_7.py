"""
Unit tests for Kernel-Density Flight Distributions (Module M14c).
"""

from pathlib import Path

import pytest

from uav_trajectory_rl.evaluation.figures_7 import (
    generate_fig7a_uav_position_density,
    generate_fig7b_altitude_and_user_density,
)


def test_generate_fig7a_creates_file(tmp_path: Path):
    """
    Verify generate_fig7a_uav_position_density creates a non-empty PNG image.
    Uses 2 seeds for fast test execution.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    fig_path = generate_fig7a_uav_position_density(
        seeds=range(2),
        cache_dir=cache_dir,
        output_dir=output_dir,
        k=10,
    )

    assert fig_path.exists()
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000
    assert fig_path.name == "fig7a_uav_position_density.png"


def test_generate_fig7b_creates_file(tmp_path: Path):
    """
    Verify generate_fig7b_altitude_and_user_density creates a non-empty PNG image.
    Uses 2 seeds for fast test execution.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    fig_path = generate_fig7b_altitude_and_user_density(
        seeds=range(2),
        cache_dir=cache_dir,
        output_dir=output_dir,
        k=10,
        user_reference_method="TDPK",
    )

    assert fig_path.exists()
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000
    assert fig_path.name == "fig7b_altitude_and_user_density.png"


def test_degenerate_covariance_fallback_does_not_crash(tmp_path: Path):
    """
    Confirm the degenerate-covariance fallback executes cleanly on PKTD3-TD's real
    checkpoint without raising LinAlgError or crashing.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    # Running generate_fig7a on seed 0 directly tests PKTD3-TD's degenerate position density
    fig_path = generate_fig7a_uav_position_density(
        seeds=[0],
        cache_dir=cache_dir,
        output_dir=output_dir,
        k=10,
    )
    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000
