"""
Sweep vs. Number of Users (Module M14d, Paper Fig. 8 Analog).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.

This module generates:
    Fig. 8 analog: 4-panel performance sweep vs. number of users k:
        (a) Average LoS Probability vs. k
        (b) Average Throughput (bps) vs. k
        (c) Average Energy Consumption (J/step) vs. k
        (d) Average DTE (throughput-energy tradeoff) vs. k

Design Conventions:
    - TDPK and Greedy are swept across k in range(10, 21) with +/- 1 std shaded error bands.
    - Dueling DQL, PPO, and PKTD3-TD checkpoints are locked to k=10 (state_dim=26);
      they appear as horizontal dashed reference lines labeled "(ref, k=10 only, not retrained)".
    - DTE is defined as: W1 * mean_throughput - W2 * (total_energy / steps_taken).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_trajectory_rl.config import W1, W2
from uav_trajectory_rl.evaluation.figures_6 import DEFAULT_METHODS, METHOD_COLORS
from uav_trajectory_rl.evaluation.harness import (
    EpisodeLog,
    MethodSpec,
    get_method_specs,
    run_batch,
)

SWEPT_METHODS: Sequence[str] = ("TDPK", "Greedy")
REFERENCE_METHODS: Sequence[str] = ("DuelingDQL", "PPO", "PKTD3-TD")


def _compute_dte(log: EpisodeLog) -> float:
    """
    Compute Decision-Tradeoff Evaluation (DTE) metric for an episode.

    Formula (Design Decision #4):
        DTE = W1 * mean_throughput - W2 * (total_energy / steps_taken)
    where W1 and W2 are imported directly from config.py.
    """
    mean_throughput = float(log.mean_throughput)
    steps = max(1, log.steps_taken)
    mean_energy_per_step = float(log.total_energy) / steps
    return float(W1 * mean_throughput - W2 * mean_energy_per_step)


def _aggregate_metrics(
    method_name: str,
    k: int,
    seeds: Sequence[int],
    cache_dir: str | Path = "results/m14_cache",
    method_spec: Optional[MethodSpec] = None,
) -> Dict[str, float]:
    """
    Run or load a batch of episodes and compute mean and std across seeds for:
        - LoS Probability
        - Transmission Rate / Throughput (bps)
        - Energy Consumption per step (J/step)
        - DTE

    Returns dict with keys:
        los_mean, los_std, throughput_mean, throughput_std,
        energy_mean, energy_std, dte_mean, dte_std
    """
    logs = run_batch(
        method_name=method_name,
        seeds=seeds,
        k=k,
        cache_dir=cache_dir,
        method_spec=method_spec,
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


def generate_fig8_user_sweep(
    k_values: Sequence[int] = range(10, 21),
    sweep_seeds: Sequence[int] = range(5),
    reference_seeds: Sequence[int] = range(5),
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Generate Fig. 8 analog: Performance metrics vs. number of ground users k.

    Produces a 2x2 subplot figure:
        (a) Average LoS Probability vs. k
        (b) Average Throughput (bps) vs. k
        (c) Average Energy Consumption (J/step) vs. k
        (d) Average DTE vs. k

    Parameters:
        k_values: Ground user counts to sweep for TDPK and Greedy (default: 10..20).
        sweep_seeds: Seeds to evaluate for each swept point (default: range(5)).
        reference_seeds: Seeds to evaluate for reference methods at k=10 (default: range(5)).
        cache_dir: Directory where per-episode logs are stored.
        output_dir: Destination directory for the generated figure.

    Returns:
        Path: Absolute path to the saved figure (fig8_user_sweep.png).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    k_list = list(k_values)
    base_specs = get_method_specs(k=10)

    # 1. Aggregate swept methods (TDPK, Greedy) across all k in k_values
    swept_data: Dict[str, Dict[str, np.ndarray]] = {}
    for method_name in SWEPT_METHODS:
        spec = base_specs[method_name]
        los_means, los_stds = [], []
        tp_means, tp_stds = [], []
        en_means, en_stds = [], []
        dte_means, dte_stds = [], []

        for k in k_list:
            agg = _aggregate_metrics(
                method_name,
                k=k,
                seeds=sweep_seeds,
                cache_dir=cache_dir,
                method_spec=spec,
            )
            los_means.append(agg["los_mean"])
            los_stds.append(agg["los_std"])
            tp_means.append(agg["throughput_mean"])
            tp_stds.append(agg["throughput_std"])
            en_means.append(agg["energy_mean"])
            en_stds.append(agg["energy_std"])
            dte_means.append(agg["dte_mean"])
            dte_stds.append(agg["dte_std"])

        swept_data[method_name] = {
            "los_mean": np.array(los_means),
            "los_std": np.array(los_stds),
            "throughput_mean": np.array(tp_means),
            "throughput_std": np.array(tp_stds),
            "energy_mean": np.array(en_means),
            "energy_std": np.array(en_stds),
            "dte_mean": np.array(dte_means),
            "dte_std": np.array(dte_stds),
        }

    # 2. Aggregate reference methods at k=10
    ref_data: Dict[str, Dict[str, float]] = {}
    for method_name in REFERENCE_METHODS:
        spec = base_specs[method_name]
        ref_data[method_name] = _aggregate_metrics(
            method_name,
            k=10,
            seeds=reference_seeds,
            cache_dir=cache_dir,
            method_spec=spec,
        )

    # 3. Create 2x2 Subplot Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    (ax_los, ax_tp), (ax_energy, ax_dte) = axes

    panels = [
        (ax_los, "los", "(a) Average LoS Probability", "Average LoS Probability", [0.0, 1.05], False),
        (ax_tp, "throughput", "(b) Average Throughput", "Average Throughput (bps)", None, True),
        (ax_energy, "energy", "(c) Average Energy Consumption", "Energy per Step (J/step)", None, False),
        (ax_dte, "dte", "(d) Average DTE Metric", "Decision-Tradeoff Evaluation (DTE)", None, False),
    ]

    for ax, metric_prefix, title, ylabel, ylim, use_sci in panels:
        # Plot swept methods (solid lines with error bands)
        for method_name in SWEPT_METHODS:
            color = METHOD_COLORS.get(method_name, None)
            mean = swept_data[method_name][f"{metric_prefix}_mean"]
            std = swept_data[method_name][f"{metric_prefix}_std"]

            ax.plot(
                k_list,
                mean,
                marker="o",
                markersize=4.5,
                color=color,
                linewidth=2.0,
                label=method_name,
            )
            ax.fill_between(
                k_list,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
            )

        # Plot reference methods (dashed horizontal lines)
        for method_name in REFERENCE_METHODS:
            color = METHOD_COLORS.get(method_name, None)
            val = ref_data[method_name][f"{metric_prefix}_mean"]

            ax.axhline(
                val,
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"{method_name} (ref, k=10 only, not retrained)",
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Users (k)", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xticks(k_list)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ylim is not None:
            ax.set_ylim(ylim)
        if use_sci:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if hasattr(ax.yaxis.get_major_formatter(), "set_useMathText"):
                ax.yaxis.get_major_formatter().set_useMathText(True)

        ax.legend(loc="best", fontsize=8.2)

    fig.suptitle(
        "Fig. 8: Performance vs. Number of Users (TDPK/Greedy swept; PPO/DuelingDQL/PKTD3-TD reference @ k=10)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_file = out_path / "fig8_user_sweep.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_file.resolve()
