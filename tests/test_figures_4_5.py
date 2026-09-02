"""
Unit tests for Trajectory and Time-Slot Snapshot Figures (Module M14a).
"""

from pathlib import Path

import pytest

from uav_trajectory_rl.evaluation.figures_4_5 import (
    generate_fig4_trajectories,
    generate_fig5_snapshots,
)


def test_generate_fig4_trajectories_creates_file(tmp_path: Path):
    """
    Verify generate_fig4_trajectories creates a non-empty PNG image.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    fig_path = generate_fig4_trajectories(
        cache_dir=cache_dir,
        output_dir=output_dir,
        seed=0,
        k=10,
    )

    assert fig_path.exists()
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000  # Non-trivial image file
    assert fig_path.name == "fig4_trajectories_comparison.png"


@pytest.mark.parametrize("method_name", ["TDPK", "PKTD3-TD"])
def test_generate_fig5_snapshots_creates_file(tmp_path: Path, method_name: str):
    """
    Verify generate_fig5_snapshots creates a non-empty PNG image for both
    working and non-convergent methods.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    fig_path = generate_fig5_snapshots(
        method_name=method_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        seed=0,
        k=10,
    )

    assert fig_path.exists()
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000
    assert fig_path.name == f"fig5_snapshots_{method_name.lower()}.png"
