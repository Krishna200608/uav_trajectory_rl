"""
Real-Time LoS Probability & Transmission Rate Curves (Module M14b, Paper Fig. 6 Analog).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.

This module generates:
    Fig. 6 analog: Two side-by-side subplots showing real-time LoS probability
    and transmission rate across time slots for all 5 methods (TDPK, Greedy,
    Dueling DQL, PPO, PKTD3-TD) at seed 0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_trajectory_rl.evaluation.harness import run_batch

DEFAULT_METHODS: Sequence[str] = ("TDPK", "Greedy", "DuelingDQL", "PPO", "PKTD3-TD")

# Centralized consistent color palette for all M14 figure suites
METHOD_COLORS: Dict[str, str] = {
    "TDPK": "#1f77b4",        # Blue
    "Greedy": "#2ca02c",      # Green
    "DuelingDQL": "#9467bd",  # Purple
    "PPO": "#ff7f0e",         # Orange
    "PKTD3-TD": "#d62728",    # Red
}


def generate_fig6_realtime_curves(
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/figures",
    seed: int = 0,
    k: int = 10,
) -> Path:
    """
    Generate Fig. 6 analog: Real-time LoS probability and transmission rate curves.

    Produces a single figure with two side-by-side subplots comparing all 5 methods:
        (a) LoS Probability vs. Time Slot
        (b) Transmission Rate (bps) vs. Time Slot

    Parameters:
        cache_dir: Directory where simulation logs are cached.
        output_dir: Destination directory for the generated figure.
        seed: Random seed for the representative flight (default: 0).
        k: Ground user count (default: 10).

    Returns:
        Path: Absolute path to the saved figure (fig6_realtime_curves.png).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, (ax_los, ax_rate) = plt.subplots(1, 2, figsize=(16, 6))

    for method_name in DEFAULT_METHODS:
        log = run_batch(
            method_name=method_name,
            seeds=[seed],
            k=k,
            cache_dir=cache_dir,
        )[0]

        steps = np.arange(log.steps_taken)
        color = METHOD_COLORS.get(method_name, None)

        # (a) Left Subplot: LoS Probability
        ax_los.plot(
            steps,
            log.los_probabilities,
            label=f"{method_name} ({log.steps_taken} steps)",
            color=color,
            linewidth=1.8,
        )

        # (b) Right Subplot: Transmission Rate (bps)
        ax_rate.plot(
            steps,
            log.transmission_rates_bps,
            label=f"{method_name} ({log.steps_taken} steps)",
            color=color,
            linewidth=1.8,
        )

    # Styling for (a) LoS Probability
    ax_los.set_title("(a) Real-Time LoS Probability", fontsize=13, fontweight="bold")
    ax_los.set_xlabel("Time Slot", fontsize=11)
    ax_los.set_ylabel("LoS Probability", fontsize=11)
    ax_los.set_ylim([0.0, 1.05])
    ax_los.grid(True, linestyle=":", alpha=0.6)
    ax_los.legend(loc="best", fontsize=9.5)

    # Styling for (b) Transmission Rate
    ax_rate.set_title("(b) Real-Time Transmission Rate", fontsize=13, fontweight="bold")
    ax_rate.set_xlabel("Time Slot", fontsize=11)
    ax_rate.set_ylabel("Transmission Rate (bps)", fontsize=11)
    ax_rate.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if hasattr(ax_rate.yaxis.get_major_formatter(), "set_useMathText"):
        ax_rate.yaxis.get_major_formatter().set_useMathText(True)
    ax_rate.grid(True, linestyle=":", alpha=0.6)
    ax_rate.legend(loc="best", fontsize=9.5)

    fig.suptitle(
        "Fig. 6: Real-Time LoS Probability & Transmission Rate (All 5 Methods, Seed 0)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_file = out_path / "fig6_realtime_curves.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_file.resolve()
