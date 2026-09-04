"""
Consolidated Comparison Table Generation (Module M14f).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.

This module evaluates all 5 methods under the reference configuration (k=10, 10 seeds)
and produces a consolidated comparison table exported to:
    - Markdown: results/tables/summary_table.md
    - CSV: results/tables/summary_table.csv

Row order follows DEFAULT_METHODS:
    1. TDPK
    2. Greedy
    3. DuelingDQL
    4. PPO
    5. PKTD3-TD

Behavior descriptions are faithfully paraphrased from the tracker sections (M10-M13, M14).
DTE formula is imported from figures_8._compute_dte.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from uav_trajectory_rl.evaluation.figures_6 import DEFAULT_METHODS
from uav_trajectory_rl.evaluation.figures_8 import _compute_dte
from uav_trajectory_rl.evaluation.harness import run_batch

# Faithful one-line paraphrases of established mechanisms from docs/PKTD3-TD_Tracker.md
METHOD_BEHAVIOR: Dict[str, str] = {
    "TDPK": "Direct-line geometric flight toward destination; ignores ground users entirely",
    "Greedy": "1-step lookahead; stalls near destination to avoid early-termination penalty until step 199",
    "DuelingDQL": "Discrete Q-learning; settles mid-field to maximize throughput without reaching destination",
    "PPO": "Continuous actor-critic; progresses ~700m+ but freezes at eastern boundary wall from azimuth bias",
    "PKTD3-TD": "Documented non-convergence (flat value surface); actor saturates and remains locked near Q_START",
}


def generate_summary_table(
    seeds: Sequence[int] = range(10),
    k: int = 10,
    cache_dir: str | Path = "results/m14_cache",
    output_dir: str | Path = "results/tables",
) -> Dict[str, Path]:
    """
    Generate a consolidated comparison table across all 5 evaluation methods.

    Evaluates or loads cached episode runs across the specified seeds and computes:
        - Arrival Rate: count / total (%)
        - Mean Steps (mean +/- std)
        - Min-Dist-to-Goal (m) (mean +/- std)
        - LoS Probability (mean +/- std)
        - Throughput (Mbps) (mean +/- std)
        - Energy (J/step) (mean +/- std)
        - DTE (mean +/- std)
        - Behavior description

    Outputs:
        results/tables/summary_table.md (Markdown pipe table)
        results/tables/summary_table.csv (CSV with separate numeric mean and std columns)

    Parameters:
        seeds: Sequence of seeds to evaluate (default: range(10)).
        k: Ground user count (default: 10).
        cache_dir: Directory where per-episode logs are cached.
        output_dir: Destination directory for summary table files.

    Returns:
        Dict[str, Path]: {"markdown": Path, "csv": Path}
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for method_name in DEFAULT_METHODS:
        logs = run_batch(
            method_name=method_name,
            seeds=seeds,
            k=k,
            cache_dir=cache_dir,
        )

        arrived = [bool(log.arrived) for log in logs]
        arr_count = sum(arrived)
        total_ep = len(logs)
        arr_pct = (arr_count / total_ep * 100.0) if total_ep > 0 else 0.0

        steps = [float(log.steps_taken) for log in logs]
        min_dist = [float(log.min_dist_to_end) for log in logs]
        los = [float(log.mean_los_probability) for log in logs]
        throughput_mbps = [float(log.mean_throughput) / 1e6 for log in logs]
        energy = [float(log.total_energy) / max(1, log.steps_taken) for log in logs]
        dte = [_compute_dte(log) for log in logs]

        behavior = METHOD_BEHAVIOR.get(method_name, "")

        rows.append({
            "method": method_name,
            "arrival_count": arr_count,
            "total_episodes": total_ep,
            "arrival_rate": float(np.mean(arrived)) if total_ep > 0 else 0.0,
            "arrival_rate_str": f"{arr_count}/{total_ep} ({arr_pct:.0f}%)",
            "steps_mean": float(np.mean(steps)),
            "steps_std": float(np.std(steps)),
            "min_dist_mean": float(np.mean(min_dist)),
            "min_dist_std": float(np.std(min_dist)),
            "los_mean": float(np.mean(los)),
            "los_std": float(np.std(los)),
            "throughput_mean": float(np.mean(throughput_mbps)),
            "throughput_std": float(np.std(throughput_mbps)),
            "energy_mean": float(np.mean(energy)),
            "energy_std": float(np.std(energy)),
            "dte_mean": float(np.mean(dte)),
            "dte_std": float(np.std(dte)),
            "behavior": behavior,
        })

    # 1. Build Markdown Table
    md_lines = [
        "# Consolidated Comparison Table (All 5 Methods, k=10, 10 Seeds)",
        "",
        "| Method | Arrival Rate | Mean Steps | Min-Dist-to-Goal (m) | LoS Probability | Throughput (Mbps) | Energy (J/step) | DTE | Behavior |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        md_lines.append(
            f"| {r['method']} "
            f"| {r['arrival_rate_str']} "
            f"| {r['steps_mean']:.2f} ± {r['steps_std']:.2f} "
            f"| {r['min_dist_mean']:.2f} ± {r['min_dist_std']:.2f} "
            f"| {r['los_mean']:.2f} ± {r['los_std']:.2f} "
            f"| {r['throughput_mean']:.2f} ± {r['throughput_std']:.2f} "
            f"| {r['energy_mean']:.2f} ± {r['energy_std']:.2f} "
            f"| {r['dte_mean']:.2f} ± {r['dte_std']:.2f} "
            f"| {r['behavior']} |"
        )

    md_lines.append("")
    md_file = out_path / "summary_table.md"
    md_file.write_text("\n".join(md_lines), encoding="utf-8")

    # 2. Build CSV File with separate numeric columns
    csv_file = out_path / "summary_table.csv"
    fieldnames = [
        "method",
        "arrival_rate_str",
        "arrival_count",
        "total_episodes",
        "arrival_rate",
        "steps_mean",
        "steps_std",
        "min_dist_to_goal_mean",
        "min_dist_to_goal_std",
        "los_mean",
        "los_std",
        "throughput_mbps_mean",
        "throughput_mbps_std",
        "energy_mean",
        "energy_std",
        "dte_mean",
        "dte_std",
        "behavior",
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "method": r["method"],
                "arrival_rate_str": r["arrival_rate_str"],
                "arrival_count": r["arrival_count"],
                "total_episodes": r["total_episodes"],
                "arrival_rate": round(r["arrival_rate"], 4),
                "steps_mean": round(r["steps_mean"], 4),
                "steps_std": round(r["steps_std"], 4),
                "min_dist_to_goal_mean": round(r["min_dist_mean"], 4),
                "min_dist_to_goal_std": round(r["min_dist_std"], 4),
                "los_mean": round(r["los_mean"], 4),
                "los_std": round(r["los_std"], 4),
                "throughput_mbps_mean": round(r["throughput_mean"], 4),
                "throughput_mbps_std": round(r["throughput_std"], 4),
                "energy_mean": round(r["energy_mean"], 4),
                "energy_std": round(r["energy_std"], 4),
                "dte_mean": round(r["dte_mean"], 4),
                "dte_std": round(r["dte_std"], 4),
                "behavior": r["behavior"],
            })

    return {
        "markdown": md_file.resolve(),
        "csv": csv_file.resolve(),
    }
