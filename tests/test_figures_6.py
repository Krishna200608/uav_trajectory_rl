"""
Unit tests for Real-Time LoS Probability & Transmission Rate Curves (Module M14b).
"""

from pathlib import Path

import pytest

from uav_trajectory_rl.evaluation.figures_6 import generate_fig6_realtime_curves


def test_generate_fig6_realtime_curves_creates_file(tmp_path: Path):
    """
    Verify generate_fig6_realtime_curves creates a non-empty PNG image.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    fig_path = generate_fig6_realtime_curves(
        cache_dir=cache_dir,
        output_dir=output_dir,
        seed=0,
        k=10,
    )

    assert fig_path.exists()
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000
    assert fig_path.name == "fig6_realtime_curves.png"
