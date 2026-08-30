"""
Greedy Baseline Diagnostic Evaluation Script (M13).

Evaluates run_greedy_episode() across 20 deterministic seeds (k=10), reporting:
  - Mean / median max displacement (m)
  - Fraction of episodes > 50m max displacement
  - Arrival rate (fraction reaching Q_END within ARRIVAL_THRESHOLD_M)
  - Mean cumulative reward
  - Wall-clock time per episode (important for M14 planning, given the
    200 deep-copy-per-step cost)

Protocol matches every other baseline exactly, for direct comparability.

Usage:
    python scripts/diagnose_greedy.py [--k K] [--n-seeds N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse

import numpy as np

from uav_trajectory_rl.baselines.greedy import run_greedy_episode
from uav_trajectory_rl.config import Q_START
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def main(
    n_eval_seeds: int = 20,
    k_users: int = 10,
) -> None:
    print("=" * 70)
    print("GREEDY BASELINE DIAGNOSTIC (M13)")
    print(f"Seeds: {n_eval_seeds} | K: {k_users} | Actions: 200 (discrete grid lookahead)")
    print("=" * 70)

    max_disps = []
    rewards = []
    arrivals = 0
    episode_times = []
    steps_list = []
    arrived_steps = []
    last_5_steps_arrivals = 0

    print(f"\n{'Seed':>5} {'Steps':>6} {'MaxDisp':>10} {'Reward':>12} {'Arrived':>8} {'Time(s)':>10}")
    print("-" * 60)

    for seed in range(n_eval_seeds):
        rng = np.random.default_rng(seed)
        env = UAVTrajectoryEnv(k=k_users, rng=rng)

        t_start = time.perf_counter()
        result = run_greedy_episode(env)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        episode_times.append(elapsed)
        steps = result["steps_taken"]
        steps_list.append(steps)

        # Max displacement from Q_START
        traj = result["trajectory"]
        disps = [
            float(np.linalg.norm(traj[i] - np.array(Q_START)))
            for i in range(len(traj))
        ]
        max_disp = max(disps)
        max_disps.append(max_disp)
        rewards.append(result["episode_reward"])
        if result["arrived"]:
            arrivals += 1
            arrived_steps.append(steps)
            if steps >= 196:  # final 5 steps of 200 (1-indexed steps 196..200, i.e. t >= 195)
                last_5_steps_arrivals += 1

        print(
            f"{seed:>5} {steps:>6d} {max_disp:>10.1f} {result['episode_reward']:>+12.2f} "
            f"{'YES' if result['arrived'] else 'NO':>8} {elapsed:>10.2f}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY (20 seeds)")
    print("=" * 70)
    print(f"Mean Max Displacement: {np.mean(max_disps):.1f} m")
    print(f"Median Max Displacement: {np.median(max_disps):.1f} m")
    print(f"Frac > 50m: {np.mean([d > 50.0 for d in max_disps]):.1%}")
    print(f"Arrival Rate: {arrivals / n_eval_seeds:.1%} ({arrivals}/{n_eval_seeds} seeds)")
    print(f"Mean Reward: {np.mean(rewards):+.2f}")
    print(f"Mean Episode Time: {np.mean(episode_times):.2f}s")
    print(f"Median Episode Time: {np.median(episode_times):.2f}s")
    print(f"Total Evaluation Time: {sum(episode_times):.1f}s")
    print("-" * 70)
    print("STEPS-TAKEN DISTRIBUTION & ARRIVAL MECHANISM:")
    print(f"Steps Taken: mean={np.mean(steps_list):.1f}, median={np.median(steps_list):.1f}, min={np.min(steps_list)}, max={np.max(steps_list)}")
    print(f"Arrivals in final 5 steps (t >= 195, step >= 196): {last_5_steps_arrivals}/{arrivals} ({last_5_steps_arrivals/max(1, arrivals):.1%})")
    print(f"Arrivals earlier than step 196: {arrivals - last_5_steps_arrivals}/{arrivals}")
    print("=" * 70)

    # Save results
    results = {
        "n_eval_seeds": n_eval_seeds,
        "k_users": k_users,
        "mean_max_disp": float(np.mean(max_disps)),
        "median_max_disp": float(np.median(max_disps)),
        "frac_gt50m": float(np.mean([d > 50.0 for d in max_disps])),
        "arrival_rate": arrivals / n_eval_seeds,
        "mean_reward": float(np.mean(rewards)),
        "steps_taken": {
            "mean": float(np.mean(steps_list)),
            "median": float(np.median(steps_list)),
            "min": int(np.min(steps_list)),
            "max": int(np.max(steps_list)),
        },
        "arrival_mechanism": {
            "total_arrivals": arrivals,
            "arrivals_in_last_5_steps": last_5_steps_arrivals,
            "arrivals_earlier": arrivals - last_5_steps_arrivals,
            "pct_last_second": float(last_5_steps_arrivals / max(1, arrivals)),
        },
        "mean_episode_time_s": float(np.mean(episode_times)),
        "median_episode_time_s": float(np.median(episode_times)),
        "total_eval_time_s": float(sum(episode_times)),
    }
    out_path = _repo_root / "checkpoints" / "greedy_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy Baseline Diagnostic (M13)")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of evaluation seeds (default: 20)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of ground users (default: 10)")
    args = parser.parse_args()
    main(n_eval_seeds=args.n_seeds, k_users=args.k)
