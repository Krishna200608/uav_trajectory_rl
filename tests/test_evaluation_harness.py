"""
Unit tests for the Shared Evaluation Harness (Module M14-Core).
"""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from uav_trajectory_rl.config import V_MAX
from uav_trajectory_rl.evaluation.harness import (
    EpisodeLog,
    MethodSpec,
    get_method_specs,
    run_batch,
    run_episode,
)
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def test_get_method_specs_all_five_methods():
    """
    Verify get_method_specs() returns all 5 methods and each loads properly on CPU.
    """
    specs = get_method_specs(device="cpu")
    expected_names = {"TDPK", "Greedy", "DuelingDQL", "PPO", "PKTD3-TD"}
    assert set(specs.keys()) == expected_names

    # Check requirements
    assert not specs["TDPK"].requires_checkpoint
    assert not specs["Greedy"].requires_checkpoint
    assert specs["DuelingDQL"].requires_checkpoint
    assert specs["PPO"].requires_checkpoint
    assert specs["PKTD3-TD"].requires_checkpoint

    # Verify each method's action_fn produces valid physical actions on a sample state
    env = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0))
    state = env.reset()

    for name, spec in specs.items():
        v, lam, rho = spec.action_fn(state, env)
        assert -1e-5 <= v <= V_MAX + 1e-5, f"{name} produced invalid speed v={v}"
        assert -1e-5 <= lam <= math.pi + 1e-5, f"{name} produced invalid polar angle lam={lam}"
        assert -math.pi - 1e-5 <= rho <= math.pi + 1e-5, f"{name} produced invalid azimuth rho={rho}"


def test_run_episode_tdpk():
    """
    Verify TDPK arrives at destination in ~89 steps under standard evaluation seed 0.
    """
    specs = get_method_specs(device="cpu")
    tdpk_spec = specs["TDPK"]

    log = run_episode(method=tdpk_spec, seed=0, k=10)

    assert log.method_name == "TDPK"
    assert log.arrived is True
    # TDPK typically arrives in 80-100 steps
    assert 75 <= log.steps_taken <= 110
    assert log.min_dist_to_end <= 5.0
    assert log.max_displacement >= 840.0
    assert len(log.positions) == log.steps_taken
    assert len(log.actions) == log.steps_taken
    assert len(log.rewards) == log.steps_taken
    assert len(log.energy_consumptions_j) == log.steps_taken


def test_run_episode_greedy():
    """
    Verify Greedy arrives at step 200 (step index 199) with boundary stall behavior.
    """
    specs = get_method_specs(device="cpu")
    greedy_spec = specs["Greedy"]

    log = run_episode(method=greedy_spec, seed=0, k=10)

    assert log.method_name == "Greedy"
    assert log.arrived is True
    assert log.steps_taken == 200
    assert log.min_dist_to_end <= 5.0
    assert log.cancellation_rate_last50 >= 0.80  # Stalled near boundary before last-step dash


def test_episode_log_npz_roundtrip(tmp_path: Path):
    """
    Verify EpisodeLog save_npz and from_npz serialization roundtrip without loss.
    """
    specs = get_method_specs(device="cpu")
    tdpk_spec = specs["TDPK"]

    log_orig = run_episode(method=tdpk_spec, seed=42, k=10)
    save_file = tmp_path / "test_tdpk_log.npz"

    log_orig.save_npz(save_file)
    assert save_file.exists()

    log_loaded = EpisodeLog.from_npz(save_file)

    assert log_loaded.method_name == log_orig.method_name
    assert log_loaded.seed == log_orig.seed
    assert log_loaded.k == log_orig.k
    assert log_loaded.user_v_init_range == log_orig.user_v_init_range
    assert log_loaded.steps_taken == log_orig.steps_taken
    assert log_loaded.arrived == log_orig.arrived
    assert np.allclose(log_loaded.positions, log_orig.positions)
    assert np.allclose(log_loaded.actions, log_orig.actions)
    assert np.allclose(log_loaded.rewards, log_orig.rewards)
    assert np.allclose(log_loaded.distances_to_end, log_orig.distances_to_end)
    assert np.allclose(log_loaded.los_probabilities, log_orig.los_probabilities)
    assert np.allclose(log_loaded.transmission_rates_bps, log_orig.transmission_rates_bps)
    assert np.allclose(log_loaded.energy_consumptions_j, log_orig.energy_consumptions_j)
    assert np.array_equal(log_loaded.position_cancelled, log_orig.position_cancelled)
    assert np.array_equal(log_loaded.arrived_flags, log_orig.arrived_flags)
    assert math.isclose(log_loaded.total_reward, log_orig.total_reward)
    assert math.isclose(log_loaded.total_energy, log_orig.total_energy)
    assert math.isclose(log_loaded.max_displacement, log_orig.max_displacement)


def test_run_batch_and_caching(tmp_path: Path):
    """
    Verify run_batch saves to disk and hits the cache on subsequent calls.
    """
    cache_dir = tmp_path / "cache"

    # First call: simulates and creates cache files
    logs_1 = run_batch(
        method_name="TDPK",
        seeds=[0, 1],
        k=10,
        cache_dir=cache_dir,
    )
    assert len(logs_1) == 2
    cached_files = list(cache_dir.glob("*.npz"))
    assert len(cached_files) == 2

    # Record file modification times
    mtimes = {f: f.stat().st_mtime_ns for f in cached_files}

    # Second call without force: must hit cache and not overwrite
    logs_2 = run_batch(
        method_name="TDPK",
        seeds=[0, 1],
        k=10,
        cache_dir=cache_dir,
        force=False,
    )
    assert len(logs_2) == 2
    assert math.isclose(logs_1[0].total_reward, logs_2[0].total_reward)
    assert math.isclose(logs_1[1].total_reward, logs_2[1].total_reward)

    # Check that cache files were not overwritten
    for f in cached_files:
        assert f.stat().st_mtime_ns == mtimes[f]
