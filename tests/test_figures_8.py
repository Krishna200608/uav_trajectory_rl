"""
Unit tests for Sweep vs. Number of Users (Module M14d).
"""

from pathlib import Path

import numpy as np
import pytest

from uav_trajectory_rl.config import W1, W2
from uav_trajectory_rl.evaluation.figures_8 import _compute_dte, generate_fig8_user_sweep
from uav_trajectory_rl.evaluation.harness import run_episode


def test_generate_fig8_creates_file(tmp_path: Path):
    """
    Verify generate_fig8_user_sweep creates a non-empty PNG image.
    Uses small parameter range (k in 10..12, 2 seeds) for test execution speed.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "figures"

    fig_path = generate_fig8_user_sweep(
        k_values=range(10, 13),
        sweep_seeds=range(2),
        reference_seeds=range(2),
        cache_dir=cache_dir,
        output_dir=output_dir,
    )

    assert fig_path.exists()
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 1000
    assert fig_path.name == "fig8_user_sweep.png"


def test_compute_dte_matches_manual_calculation():
    """
    Verify _compute_dte matches a manual calculation using raw EpisodeLog fields
    and config.W1 / config.W2.
    """
    from uav_trajectory_rl.evaluation.harness import get_method_specs
    specs = get_method_specs(k=10)
    log = run_episode(specs["TDPK"], seed=0, k=10)

    # 1. Computed via helper
    val_helper = _compute_dte(log)

    # 2. Hand-derived calculation
    mean_tp = float(log.mean_throughput)
    mean_energy = float(log.total_energy) / float(log.steps_taken)
    val_manual = float(W1 * mean_tp - W2 * mean_energy)

    print(f"DTE Helper: {val_helper}, DTE Manual: {val_manual}")
    assert np.isclose(val_helper, val_manual, rtol=1e-7, atol=1e-7)
