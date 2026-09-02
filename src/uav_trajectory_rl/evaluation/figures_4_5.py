"""
Trajectory and Time-Slot Snapshot Figures (Module M14a, Paper Figs. 4-5 Analogs).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.

This module generates:
    1. Fig. 4 analog: 3-D flight trajectory comparison across all 5 methods
       (TDPK, Greedy, Dueling DQL, PPO, PKTD3-TD) in a combined 5-subplot figure.
    2. Fig. 5 analog: 2-D time-slot snapshots (UAV and ground user positions at 6
       time slots, with z altitude and LoS probability in subplot titles) for each method.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import numpy as np

from uav_trajectory_rl.config import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
)
from uav_trajectory_rl.evaluation.harness import run_batch

DEFAULT_METHODS: Sequence[str] = ("TDPK", "Greedy", "DuelingDQL", "PPO", "PKTD3-TD")


def generate_fig4_trajectories(
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/figures",
    seed: int = 0,
    k: int = 10,
) -> Path:
    """
    Generate Fig. 4 analog: 3-D flight trajectories for all 5 methods.

    Produces a single figure containing 5 3-D subplots (2x3 grid, 5 active + 1 hidden).

    Parameters:
        cache_dir: Directory where simulation logs are cached.
        output_dir: Destination directory for the generated figure.
        seed: Random seed for the representative flight (default: 0).
        k: Ground user count (default: 10).

    Returns:
        Path: Absolute path to the saved figure (fig4_trajectories_comparison.png).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 11))

    # Grid of 2 rows, 3 columns
    for idx, method_name in enumerate(DEFAULT_METHODS):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")

        # Obtain episode log via cached evaluation harness
        log = run_batch(
            method_name=method_name,
            seeds=[seed],
            k=k,
            cache_dir=cache_dir,
        )[0]

        pos = log.positions
        start_pos = log.start_pos
        target_pos = log.target_pos

        # Plot 3D trajectory line
        if len(pos) > 0:
            full_path = np.vstack([start_pos, pos])
            ax.plot(
                full_path[:, 0],
                full_path[:, 1],
                full_path[:, 2],
                color="navy",
                linewidth=2.0,
                label="UAV Trajectory",
            )
            # Final UAV position
            ax.scatter(
                pos[-1, 0],
                pos[-1, 1],
                pos[-1, 2],
                marker="*",
                color="blue",
                s=160,
                label="UAV Final Pos",
                zorder=5,
            )

        # Start and Destination markers
        ax.scatter(
            start_pos[0],
            start_pos[1],
            start_pos[2],
            marker="o",
            facecolors="none",
            edgecolors="dimgray",
            s=90,
            linewidth=2.0,
            label="Start Q_START",
            zorder=4,
        )
        ax.scatter(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            marker="X",
            color="dimgray",
            s=100,
            label="Destination Q_END",
            zorder=4,
        )

        # Fixed axis ranges for honest comparison across all methods
        ax.set_xlim([X_MIN, X_MAX])
        ax.set_ylim([Y_MIN, Y_MAX])
        ax.set_zlim([Z_MIN, Z_MAX])

        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.set_zlabel("Z (m)", fontsize=9)

        status_str = "arrived" if log.arrived else "did not arrive"
        ax.set_title(
            f"{method_name} ({status_str}, {log.steps_taken} steps)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=7.5)

    # Hide unused 6th subplot
    ax_empty = fig.add_subplot(2, 3, 6)
    ax_empty.set_visible(False)

    fig.suptitle(
        "Fig. 4: 3-D Flight Trajectory Comparison Across All 5 Methods (Seed 0)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_file = out_path / "fig4_trajectories_comparison.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_file.resolve()


def generate_fig5_snapshots(
    method_name: str,
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/figures",
    seed: int = 0,
    k: int = 10,
) -> Path:
    """
    Generate Fig. 5 analog: 2-D time-slot snapshots for a specific method.

    Produces a 2x3 grid of 2-D subplots showing UAV and mobile ground user positions
    at 6 time slots across the episode.

    Parameters:
        method_name: Method identifier ("TDPK", "Greedy", "DuelingDQL", "PPO", "PKTD3-TD").
        cache_dir: Directory where simulation logs are cached.
        output_dir: Destination directory for the generated figure.
        seed: Random seed for the representative flight (default: 0).
        k: Ground user count (default: 10).

    Returns:
        Path: Absolute path to the saved figure (fig5_snapshots_{method_name.lower()}.png).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log = run_batch(
        method_name=method_name,
        seeds=[seed],
        k=k,
        cache_dir=cache_dir,
    )[0]

    steps_taken = log.steps_taken
    pos = log.positions
    user_hist = log.user_positions_history
    start_pos = log.start_pos
    target_pos = log.target_pos

    # Compute 6 snapshot slot indices (DESIGN DECISION #3)
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    raw_slots = [int(round(f * (steps_taken - 1))) for f in fractions]
    # Deduplicate while preserving order
    unique_slots: List[int] = []
    for s in raw_slots:
        s_clamped = max(0, min(steps_taken - 1, s))
        if s_clamped not in unique_slots:
            unique_slots.append(s_clamped)

    # If deduplication produced fewer than 6 slots (e.g. very short episode), pad
    while len(unique_slots) < 6:
        last = unique_slots[-1]
        unique_slots.append(last)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for idx, slot_idx in enumerate(unique_slots[:6]):
        ax = axes_flat[idx]

        # Trajectory history up to this slot
        if slot_idx >= 0 and len(pos) > 0:
            hist_path = np.vstack([start_pos[:2], pos[: slot_idx + 1, :2]])
            ax.plot(
                hist_path[:, 0],
                hist_path[:, 1],
                color="blue",
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label="Trajectory",
            )
            # Current UAV position
            ax.scatter(
                pos[slot_idx, 0],
                pos[slot_idx, 1],
                marker="*",
                color="blue",
                s=180,
                label="UAV",
                zorder=5,
            )

        # Ground user positions at this slot
        if slot_idx < len(user_hist):
            users_xy = user_hist[slot_idx]
            ax.scatter(
                users_xy[:, 0],
                users_xy[:, 1],
                marker="^",
                color="red",
                s=70,
                label="Ground Users",
                zorder=4,
            )

        # Start and Destination markers
        ax.scatter(
            start_pos[0],
            start_pos[1],
            marker="o",
            facecolors="none",
            edgecolors="dimgray",
            s=90,
            linewidth=1.8,
            label="Start Q_START",
            zorder=3,
        )
        ax.scatter(
            target_pos[0],
            target_pos[1],
            marker="X",
            color="dimgray",
            s=100,
            label="Destination Q_END",
            zorder=3,
        )

        ax.set_xlim([X_MIN, X_MAX])
        ax.set_ylim([Y_MIN, Y_MAX])
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        z_val = pos[slot_idx, 2] if slot_idx < len(pos) else start_pos[2]
        lp_val = log.los_probabilities[slot_idx] if slot_idx < len(log.los_probabilities) else 0.0
        ax.set_title(
            f"n={slot_idx}, z={z_val:.1f}m, Lp={lp_val:.2f}",
            fontsize=11,
            fontweight="bold",
        )

        # Include clean legend on first subplot
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.5)

    clean_name = method_name.lower().replace("-", "_")
    fig.suptitle(
        f"Fig. 5: Time-Slot Snapshots for {method_name} (Seed 0)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    filename = f"fig5_snapshots_{method_name.lower()}.png"
    save_file = out_path / filename
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_file.resolve()
