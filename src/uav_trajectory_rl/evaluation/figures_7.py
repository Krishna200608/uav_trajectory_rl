"""
Kernel-Density Flight Distributions (Module M14c, Paper Fig. 7 Analog).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.

This module generates:
    1. Fig. 7(a) analog: 2-D Kernel Density Estimation (KDE) of UAV xy-positions
       for all 5 methods over 10 repeated flights (fig7a_uav_position_density.png).
    2. Fig. 7(b) analog: Overlay of 1-D altitude KDE curves for all 5 methods and
       2-D KDE of ground user positions (fig7b_altitude_and_user_density.png).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from uav_trajectory_rl.config import (
    Q_END,
    Q_START,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
)
from uav_trajectory_rl.evaluation.figures_6 import DEFAULT_METHODS, METHOD_COLORS
from uav_trajectory_rl.evaluation.harness import run_batch


def _collect_flight_data(
    method_name: str,
    seeds: Sequence[int],
    k: int = 10,
    cache_dir: str | Path = "results/m14_cache",
) -> Dict[str, Any]:
    """
    Simulate or load episodes across seeds and aggregate flattened spatial arrays.

    Parameters:
        method_name: Identifier of the trajectory method.
        seeds: Sequence of random seeds (e.g. range(10)).
        k: Ground user count.
        cache_dir: Directory where per-seed simulation logs are cached.

    Returns:
        Dict[str, Any]:
            - "xy": np.ndarray of shape (N, 2), all UAV (x, y) across all steps and flights.
            - "z": np.ndarray of shape (N,), all UAV altitudes across all steps and flights.
            - "user_xy": np.ndarray of shape (M, 2), all user (x, y) across all steps and flights.
            - "logs": list of EpisodeLog objects.
    """
    logs = run_batch(
        method_name=method_name,
        seeds=seeds,
        k=k,
        cache_dir=cache_dir,
    )

    xy_list = []
    z_list = []
    user_xy_list = []

    for log in logs:
        if len(log.positions) > 0:
            xy_list.append(log.positions[:, :2])
            z_list.append(log.positions[:, 2])
        if len(log.user_positions_history) > 0:
            # log.user_positions_history has shape (T, k, 2) -> reshape to (T*k, 2)
            user_xy_list.append(log.user_positions_history.reshape(-1, 2))

    xy = np.vstack(xy_list) if len(xy_list) > 0 else np.empty((0, 2), dtype=np.float64)
    z = np.concatenate(z_list) if len(z_list) > 0 else np.empty((0,), dtype=np.float64)
    user_xy = np.vstack(user_xy_list) if len(user_xy_list) > 0 else np.empty((0, 2), dtype=np.float64)

    return {
        "xy": xy,
        "z": z,
        "user_xy": user_xy,
        "logs": logs,
    }


def generate_fig7a_uav_position_density(
    seeds: Sequence[int] = range(10),
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/figures",
    k: int = 10,
) -> Path:
    """
    Generate Fig. 7(a) analog: 2-D Kernel Density Estimation of UAV xy-positions.

    Produces a 2x3 subplot grid (5 active + 1 hidden) comparing all 5 methods over
    10 flights. Robustly catches degenerate/singular covariance matrices (e.g. for
    non-convergent PKTD3-TD) and falls back to a scatter representation with an
    honest diagnostic label.

    Parameters:
        seeds: Sequence of random seeds to evaluate (default: range(10)).
        cache_dir: Directory where simulation logs are cached.
        output_dir: Destination directory for the generated figure.
        k: Ground user count (default: 10).

    Returns:
        Path: Absolute path to the saved figure (fig7a_uav_position_density.png).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 11))

    # Grid for density evaluation
    grid_size = 100
    x_grid = np.linspace(X_MIN, X_MAX, grid_size)
    y_grid = np.linspace(Y_MIN, Y_MAX, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    for idx, method_name in enumerate(DEFAULT_METHODS):
        ax = fig.add_subplot(2, 3, idx + 1)
        data = _collect_flight_data(method_name, seeds=seeds, k=k, cache_dir=cache_dir)
        xy = data["xy"]
        color = METHOD_COLORS.get(method_name, "navy")

        is_degenerate = False
        near_zero_variance = False
        density = None

        # Check for degenerate covariance or near-zero variance
        if len(xy) < 3 or np.any(np.var(xy, axis=0) < 1e-5):
            is_degenerate = True
            near_zero_variance = True
        else:
            try:
                kde = scipy.stats.gaussian_kde(xy.T)
                density = kde(grid_coords).reshape(xx.shape)
            except (np.linalg.LinAlgError, ValueError):
                is_degenerate = True
                near_zero_variance = False

        if not is_degenerate and density is not None:
            cs = ax.contourf(xx, yy, density, levels=15, cmap="viridis")
            plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{method_name}", fontsize=12, fontweight="bold")
        else:
            # Degenerate-covariance fallback (DESIGN DECISION #5)
            # Display actual sample positions with explicit notice rather than failing
            ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.4, color=color, label="Visited Positions")
            if near_zero_variance:
                fallback_reason = "insufficient movement for density estimate"
            else:
                fallback_reason = "near-collinear trajectory — density undefined"
            ax.set_title(
                f"{method_name} ({fallback_reason})",
                fontsize=10.0,
                fontweight="bold",
                color="darkred",
            )

        # Draw Start and Destination markers
        ax.scatter(
            Q_START[0],
            Q_START[1],
            marker="o",
            facecolors="none",
            edgecolors="dimgray",
            s=80,
            linewidth=1.8,
            label="Start Q_START",
            zorder=5,
        )
        ax.scatter(
            Q_END[0],
            Q_END[1],
            marker="X",
            color="dimgray",
            s=90,
            label="Destination Q_END",
            zorder=5,
        )

        ax.set_xlim([X_MIN, X_MAX])
        ax.set_ylim([Y_MIN, Y_MAX])
        ax.set_xlabel("X (m)", fontsize=9.5)
        ax.set_ylabel("Y (m)", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=7.5)

    # Hide 6th unused subplot
    ax_empty = fig.add_subplot(2, 3, 6)
    ax_empty.set_visible(False)

    num_flights = len(seeds)
    fig.suptitle(
        f"Fig. 7(a): Kernel Density of UAV Position ({num_flights} Flights, All 5 Methods)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_file = out_path / "fig7a_uav_position_density.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_file.resolve()


def generate_fig7b_altitude_and_user_density(
    seeds: Sequence[int] = range(10),
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/figures",
    k: int = 10,
    user_reference_method: str = "TDPK",
) -> Path:
    """
    Generate Fig. 7(b) analog: UAV altitude density overlay & ground user position density.

    Produces a 2-panel figure:
        - Left: 1-D KDE density curves of flight altitude across all 5 methods.
        - Right: 2-D KDE density contour of mobile ground user positions.

    Parameters:
        seeds: Sequence of random seeds to evaluate (default: range(10)).
        cache_dir: Directory where simulation logs are cached.
        output_dir: Destination directory for the generated figure.
        k: Ground user count (default: 10).
        user_reference_method: Reference method used to extract user mobility (default: "TDPK").

    Returns:
        Path: Absolute path to the saved figure (fig7b_altitude_and_user_density.png).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, (ax_alt, ax_user) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left Subplot: 1-D Altitude Density Overlay ---
    z_grid = np.linspace(Z_MIN, Z_MAX, 300)

    for method_name in DEFAULT_METHODS:
        data = _collect_flight_data(method_name, seeds=seeds, k=k, cache_dir=cache_dir)
        z = data["z"]
        color = METHOD_COLORS.get(method_name, None)

        if len(z) < 2 or float(np.std(z)) < 1e-5:
            # Altitude variance is near zero (e.g. horizontal flight or stationary lock-in)
            const_z = float(z[0]) if len(z) > 0 else float(Q_START[2])
            ax_alt.axvline(
                const_z,
                linestyle="--",
                color=color,
                linewidth=2.0,
                label=f"{method_name} (constant {const_z:.1f}m)",
            )
        else:
            try:
                kde_z = scipy.stats.gaussian_kde(z)
                dens_z = kde_z(z_grid)
                ax_alt.plot(
                    z_grid,
                    dens_z,
                    color=color,
                    linewidth=2.0,
                    label=method_name,
                )
            except (np.linalg.LinAlgError, ValueError):
                const_z = float(np.mean(z))
                ax_alt.axvline(
                    const_z,
                    linestyle="--",
                    color=color,
                    linewidth=2.0,
                    label=f"{method_name} (near-constant {const_z:.1f}m)",
                )

    ax_alt.set_title("(a) UAV Flight Altitude Density", fontsize=13, fontweight="bold")
    ax_alt.set_xlabel("Altitude Z (m)", fontsize=11)
    ax_alt.set_ylabel("Density", fontsize=11)
    ax_alt.set_xlim([Z_MIN, Z_MAX])
    ax_alt.grid(True, linestyle=":", alpha=0.6)
    ax_alt.legend(loc="best", fontsize=9)

    # --- Right Subplot: 2-D User Position Density ---
    ref_data = _collect_flight_data(user_reference_method, seeds=seeds, k=k, cache_dir=cache_dir)
    user_xy = ref_data["user_xy"]

    grid_size = 100
    x_grid = np.linspace(X_MIN, X_MAX, grid_size)
    y_grid = np.linspace(Y_MIN, Y_MAX, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    is_user_deg = False
    u_density = None

    if len(user_xy) < 3 or np.any(np.var(user_xy, axis=0) < 1e-5):
        is_user_deg = True
    else:
        try:
            kde_u = scipy.stats.gaussian_kde(user_xy.T)
            u_density = kde_u(grid_coords).reshape(xx.shape)
        except (np.linalg.LinAlgError, ValueError):
            is_user_deg = True

    if not is_user_deg and u_density is not None:
        cs_u = ax_user.contourf(xx, yy, u_density, levels=15, cmap="magma")
        plt.colorbar(cs_u, ax=ax_user, fraction=0.046, pad=0.04)
        ax_user.set_title(
            f"(b) User Position Density (ref: {user_reference_method}, {len(seeds)} flights)",
            fontsize=12,
            fontweight="bold",
        )
    else:
        ax_user.scatter(user_xy[:, 0], user_xy[:, 1], s=5, alpha=0.2, color="crimson")
        ax_user.set_title(
            f"(b) User Positions (ref: {user_reference_method}, {len(seeds)} flights)",
            fontsize=12,
            fontweight="bold",
        )

    ax_user.set_xlim([X_MIN, X_MAX])
    ax_user.set_ylim([Y_MIN, Y_MAX])
    ax_user.set_xlabel("X (m)", fontsize=11)
    ax_user.set_ylabel("Y (m)", fontsize=11)
    ax_user.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Fig. 7(b): UAV Altitude Density & User Position Density",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_file = out_path / "fig7b_altitude_and_user_density.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_file.resolve()
