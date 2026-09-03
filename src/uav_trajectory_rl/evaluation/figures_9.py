"""
Sweep vs. User Mobility Speed (Module M14e, Paper Fig. 9 Analog).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.

This module generates:
    Fig. 9 analog: 4-panel performance sweep vs. user mobility speed v_mob in (2, 4, 6, 8, 10, 12) m/s:
        (a) Average LoS Probability vs. v_mob
        (b) Average Throughput (bps) vs. v_mob
        (c) Average Energy Consumption (J/step) vs. v_mob
        (d) Average DTE (throughput-energy tradeoff) vs. v_mob

Design Conventions:
    - All 5 methods (TDPK, Greedy, Dueling DQL, PPO, PKTD3-TD) are swept across all 6 speed points
      with +/- 1 std shaded error bands. (Unlike Fig. 8, state_dim is invariant to user velocities,
      so checkpoint-based models can be evaluated directly).
    - Speed mapping follows M14-Core: user_v_init_range = (max(0.1, v_mob - 1.0), v_mob + 1.0).
    - Fixed k=10 ground users.
    - DTE formula is imported from figures_8._compute_dte.
    - Out-Of-Distribution (OOD) caveat: Trained models were trained at v_init_range=(0.5, 2.0);
      evaluations at speeds up to 12 m/s test OOD generalization.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_trajectory_rl.evaluation.figures_6 import DEFAULT_METHODS, METHOD_COLORS
from uav_trajectory_rl.evaluation.figures_8 import _compute_dte
from uav_trajectory_rl.evaluation.harness import run_batch


def speed_to_range(v_mob: float) -> Tuple[float, float]:
    """
    Map scalar mobility speed to a symmetric +/- 1.0 m/s velocity initialization range,
    floored at 0.1 m/s (Design Decision #2, establishing M14-Core convention).

    Parameters:
        v_mob: Target user mobility speed in m/s.

    Returns:
        Tuple[float, float]: (v_min, v_max) speed bounds.
    """
    return (max(0.1, float(v_mob) - 1.0), float(v_mob) + 1.0)


def _aggregate_speed_metrics(
    method_name: str,
    v_range: Tuple[float, float],
    seeds: Sequence[int],
    k: int = 10,
    cache_dir: str | Path = "results/m14_cache",
) -> Dict[str, float]:
    """
    Run or load a batch of episodes for a given speed range and compute mean and std across seeds.

    Returns dict with keys:
        los_mean, los_std, throughput_mean, throughput_std,
        energy_mean, energy_std, dte_mean, dte_std
    """
    logs = run_batch(
        method_name=method_name,
        seeds=seeds,
        k=k,
        user_v_init_range=v_range,
        cache_dir=cache_dir,
    )

    los_list = [float(log.mean_los_probability) for log in logs]
    tp_list = [float(log.mean_throughput) for log in logs]
    energy_list = [float(log.total_energy) / max(1, log.steps_taken) for log in logs]
    dte_list = [_compute_dte(log) for log in logs]

    return {
        "los_mean": float(np.mean(los_list)),
        "los_std": float(np.std(los_list)),
        "throughput_mean": float(np.mean(tp_list)),
        "throughput_std": float(np.std(tp_list)),
        "energy_mean": float(np.mean(energy_list)),
        "energy_std": float(np.std(energy_list)),
        "dte_mean": float(np.mean(dte_list)),
        "dte_std": float(np.std(dte_list)),
    }


def generate_fig9_speed_sweep(
    speed_values: Sequence[float | int] = (2, 4, 6, 8, 10, 12),
    sweep_seeds: Sequence[int] = range(5),
    k: int = 10,
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Generate Fig. 9 analog: Performance metrics vs. ground user mobility speed.

    Produces a 2x2 subplot figure evaluating all 5 methods across discrete speeds:
        (a) Average LoS Probability vs. User Mobility Speed
        (b) Average Throughput (bps) vs. User Mobility Speed
        (c) Average Energy Consumption (J/step) vs. User Mobility Speed
        (d) Average DTE vs. User Mobility Speed

    Parameters:
        speed_values: Discrete mobility speeds to evaluate in m/s (default: 2, 4, 6, 8, 10, 12).
        sweep_seeds: Seeds to evaluate for each speed point (default: range(5)).
        k: Ground user count (default: 10).
        cache_dir: Directory where per-episode logs are stored.
        output_dir: Destination directory for the generated figure.

    Returns:
        Path: Absolute path to the saved figure (fig9_speed_sweep.png).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    speeds = [float(s) for s in speed_values]

    # Aggregate metrics across all 5 methods and all speed values
    method_data: Dict[str, Dict[str, np.ndarray]] = {}

    for method_name in DEFAULT_METHODS:
        los_means, los_stds = [], []
        tp_means, tp_stds = [], []
        en_means, en_stds = [], []
        dte_means, dte_stds = [], []

        for v_mob in speeds:
            v_range = speed_to_range(v_mob)
            agg = _aggregate_speed_metrics(
                method_name=method_name,
                v_range=v_range,
                seeds=sweep_seeds,
                k=k,
                cache_dir=cache_dir,
            )
            los_means.append(agg["los_mean"])
            los_stds.append(agg["los_std"])
            tp_means.append(agg["throughput_mean"])
            tp_stds.append(agg["throughput_std"])
            en_means.append(agg["energy_mean"])
            en_stds.append(agg["energy_std"])
            dte_means.append(agg["dte_mean"])
            dte_stds.append(agg["dte_std"])

        method_data[method_name] = {
            "los_mean": np.array(los_means),
            "los_std": np.array(los_stds),
            "throughput_mean": np.array(tp_means),
            "throughput_std": np.array(tp_stds),
            "energy_mean": np.array(en_means),
            "energy_std": np.array(en_stds),
            "dte_mean": np.array(dte_means),
            "dte_std": np.array(dte_stds),
        }

    # Create 2x2 Subplot Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))
    (ax_los, ax_tp), (ax_energy, ax_dte) = axes

    panels = [
        (ax_los, "los", "(a) Average LoS Probability", "Average LoS Probability", [0.0, 1.05], False),
        (ax_tp, "throughput", "(b) Average Throughput", "Average Throughput (bps)", None, True),
        (ax_energy, "energy", "(c) Average Energy Consumption", "Energy per Step (J/step)", None, False),
        (ax_dte, "dte", "(d) Average DTE Metric", "Decision-Tradeoff Evaluation (DTE)", None, False),
    ]

    for ax, metric_prefix, title, ylabel, ylim, use_sci in panels:
        for method_name in DEFAULT_METHODS:
            color = METHOD_COLORS.get(method_name, None)
            mean = method_data[method_name][f"{metric_prefix}_mean"]
            std = method_data[method_name][f"{metric_prefix}_std"]

            ax.plot(
                speeds,
                mean,
                marker="o",
                markersize=5.0,
                color=color,
                linewidth=2.0,
                label=method_name,
            )
            ax.fill_between(
                speeds,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("User Mobility Speed (m/s)", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xticks(speeds)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ylim is not None:
            ax.set_ylim(ylim)
        if use_sci:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if hasattr(ax.yaxis.get_major_formatter(), "set_useMathText"):
                ax.yaxis.get_major_formatter().set_useMathText(True)

        ax.legend(loc="best", fontsize=8.5)

    # Figure Suptitle & Prominent OOD Footnote (Design Decision #5)
    fig.suptitle(
        "Fig. 9: Performance vs. User Mobility Speed (All 5 Methods, k=10)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Note: Dueling DQL, PPO, and PKTD3-TD were trained at speeds ~0.5-2.0 m/s; "
        "results beyond that range reflect out-of-distribution generalization, not additional training.",
        ha="center",
        fontsize=9.5,
        style="italic",
        color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.035, 1, 0.96])

    save_file = out_path / "fig9_speed_sweep.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_file.resolve()
