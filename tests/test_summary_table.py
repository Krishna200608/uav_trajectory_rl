"""
Unit tests for Consolidated Comparison Table Generation (Module M14f).
"""

import csv
from pathlib import Path
from typing import Dict

import pytest

from uav_trajectory_rl.evaluation.summary_table import generate_summary_table


@pytest.fixture(scope="module")
def summary_results(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """
    Generate summary table results once for the test module (3 seeds) to avoid
    redundant simulation across tests.
    """
    base_dir = tmp_path_factory.mktemp("summary_test")
    cache_dir = base_dir / "cache"
    output_dir = base_dir / "tables"

    return generate_summary_table(
        seeds=range(3),
        k=10,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


def test_generate_summary_table_creates_both_files(summary_results: Dict[str, Path]):
    """
    Verify generate_summary_table produces both Markdown and CSV files with nonzero sizes.
    """
    assert "markdown" in summary_results
    assert "csv" in summary_results

    md_path = summary_results["markdown"]
    csv_path = summary_results["csv"]

    assert md_path.exists()
    assert md_path.is_file()
    assert md_path.stat().st_size > 100
    assert md_path.name == "summary_table.md"

    assert csv_path.exists()
    assert csv_path.is_file()
    assert csv_path.stat().st_size > 100
    assert csv_path.name == "summary_table.csv"


def test_arrival_rate_known_values(summary_results: Dict[str, Path]):
    """
    Verify arrival rate regressions across known baseline behaviors:
    - TDPK and Greedy must have 100% arrival rate (3/3).
    - PKTD3-TD must have 0% arrival rate (0/3).
    """
    csv_path = summary_results["csv"]

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        by_method = {row["method"]: row for row in reader}

    assert by_method["TDPK"]["arrival_rate_str"] == "3/3 (100%)"
    assert float(by_method["TDPK"]["arrival_rate"]) == 1.0

    assert by_method["Greedy"]["arrival_rate_str"] == "3/3 (100%)"
    assert float(by_method["Greedy"]["arrival_rate"]) == 1.0

    assert by_method["PKTD3-TD"]["arrival_rate_str"] == "0/3 (0%)"
    assert float(by_method["PKTD3-TD"]["arrival_rate"]) == 0.0


def test_csv_has_separate_numeric_columns(summary_results: Dict[str, Path]):
    """
    Verify CSV has distinct, float-parseable columns for mean and std rather than packed strings.
    """
    csv_path = summary_results["csv"]

    numeric_columns = [
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
    ]

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 5  # 5 default methods

    for row in rows:
        for col in numeric_columns:
            assert col in row, f"Missing numeric column: {col}"
            val = float(row[col])
            assert not (val != val), f"Column {col} resulted in NaN"
