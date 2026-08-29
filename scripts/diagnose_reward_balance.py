"""
Reward Component Balance Diagnostic: TDPK (Full Journey) vs Always-Hover.

Quantifies the per-component reward breakdown (r1 through r6) across 30 seeds
to determine whether completing a full journey provides a large, robust learning
signal over hovering or if our assumed (non-paper-specified) parameters
(FC_HZ, N0_DBM_HZ, OMEGA) create a narrow, weak margin.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from uav_trajectory_rl.baselines.tdpk import tdpk_action
from uav_trajectory_rl.config import N_SLOTS, Q_END, Q_START, V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def run_episode_with_breakdown(
    env: UAVTrajectoryEnv,
    policy_type: str,
    rng: np.random.Generator,
) -> Dict[str, float]:
    state = env.reset()
    done = False
    step_count = 0
    arrived = False

    total_r1 = 0.0
    total_r2 = 0.0
    total_r3 = 0.0
    total_r4 = 0.0
    total_r5 = 0.0
    total_r6 = 0.0
    total_reward = 0.0

    while not done:
        if policy_type == "hover":
            action = (0.0, math.pi / 2.0, 0.0)
        elif policy_type == "tdpk":
            action = tdpk_action(
                current_pos=env.uav_pos,
                destination=env.q_end,
                v_max=env.v_max,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown policy: {policy_type}")

        next_state, reward, done, info = env.step(action)
        step_count += 1

        total_r1 += info["r1_throughput"]
        total_r2 += info["r2_energy"]
        total_r3 += info["r3_terminal"]
        total_r4 += info["r4_proximity"]
        total_r5 += info["r5_accel"]
        total_r6 += info["r6_height"]
        total_reward += reward

        if info.get("arrived", False):
            arrived = True

        state = next_state

    dist_to_end = float(np.linalg.norm(env.uav_pos - env.q_end))
    dist_from_start = float(np.linalg.norm(env.uav_pos - env.q_start))

    return {
        "arrived": arrived,
        "steps": step_count,
        "r1_throughput": total_r1,
        "r2_energy": total_r2,
        "r3_terminal": total_r3,
        "r4_proximity": total_r4,
        "r5_accel": total_r5,
        "r6_height": total_r6,
        "total_reward": total_reward,
        "final_dist_to_end": dist_to_end,
        "final_disp_from_start": dist_from_start,
    }


def evaluate_policy_breakdown(
    policy_type: str,
    seeds: List[int],
    k: int = 10,
) -> List[Dict[str, float]]:
    records = []
    for seed in seeds:
        env_rng = np.random.default_rng(seed)
        action_rng = np.random.default_rng(seed + 10000)
        env = UAVTrajectoryEnv(k=k, rng=env_rng)
        rec = run_episode_with_breakdown(env, policy_type, rng=action_rng)
        rec["seed"] = seed
        records.append(rec)
    return records


def print_summary_table(name: str, records: List[Dict[str, float]], filter_arrived: bool = False):
    if filter_arrived:
        subset = [r for r in records if r["arrived"]]
        desc = f"{name} (Arrived Episodes Only: {len(subset)}/{len(records)})"
    else:
        subset = records
        desc = f"{name} (All Episodes: {len(subset)})"

    mean_steps = np.mean([r["steps"] for r in subset])
    mean_r1 = np.mean([r["r1_throughput"] for r in subset])
    mean_r2 = np.mean([r["r2_energy"] for r in subset])
    mean_r3 = np.mean([r["r3_terminal"] for r in subset])
    mean_r4 = np.mean([r["r4_proximity"] for r in subset])
    mean_r5 = np.mean([r["r5_accel"] for r in subset])
    mean_r6 = np.mean([r["r6_height"] for r in subset])
    mean_total = np.mean([r["total_reward"] for r in subset])

    # Fractions
    energy_over_throughput = abs(mean_r2) / mean_r1 * 100.0 if mean_r1 != 0 else 0.0
    neg_total = abs(mean_r2 + mean_r3 + mean_r4 + mean_r5 + mean_r6)
    neg_over_throughput = neg_total / mean_r1 * 100.0 if mean_r1 != 0 else 0.0

    print(f"\n### {desc}")
    print(f"- Mean Steps: {mean_steps:.1f} / {N_SLOTS}")
    print(f"- r1 (Throughput):       {mean_r1:+10.2f}")
    print(f"- r2 (Energy):           {mean_r2:+10.2f}  ({energy_over_throughput:.1f}% of throughput)")
    print(f"- r3 (Terminal Arrival): {mean_r3:+10.2f}")
    print(f"- r4 (Proximity/Lack):   {mean_r4:+10.2f}")
    print(f"- r5 (Acceleration):     {mean_r5:+10.2f}")
    print(f"- r6 (Height):           {mean_r6:+10.2f}")
    print(f"- Total Reward:          {mean_total:+10.2f}")
    print(f"- Total Negative Costs:  {-neg_total:+10.2f}  ({neg_over_throughput:.1f}% of throughput)")

    return {
        "steps": mean_steps,
        "r1": mean_r1,
        "r2": mean_r2,
        "r3": mean_r3,
        "r4": mean_r4,
        "r5": mean_r5,
        "r6": mean_r6,
        "total": mean_total,
        "energy_pct_r1": energy_over_throughput,
    }


def main():
    num_seeds = 30
    seeds = list(range(num_seeds))
    k = 10

    print("=" * 80)
    print(f"REWARD COMPONENT BALANCE DIAGNOSTIC ({num_seeds} SEEDS, k={k})")
    print("Investigating whether the throughput-vs-energy reward balance makes full journeys robustly favorable.")
    print("=" * 80)

    print("\nRunning TDPK across 30 seeds...")
    tdpk_records = evaluate_policy_breakdown("tdpk", seeds, k=k)

    print("Running Always-Hover across 30 seeds...")
    hover_records = evaluate_policy_breakdown("hover", seeds, k=k)

    # Step 1: TDPK Breakdown
    tdpk_stats = print_summary_table("TDPK Direct Flight", tdpk_records, filter_arrived=True)

    # Step 2: Hover Breakdown
    hover_stats = print_summary_table("Always Hover (v=0)", hover_records, filter_arrived=False)

    # Step 3: Side-by-side comparison & advantage breakdown
    print("\n" + "=" * 80)
    print("STEP 3: SIDE-BY-SIDE REWARD BALANCE COMPARISON (TDPK vs HOVER)")
    print("=" * 80)

    delta_total = tdpk_stats["total"] - hover_stats["total"]
    ratio = tdpk_stats["total"] / hover_stats["total"] if hover_stats["total"] != 0 else float("inf")
    pct_gain = (ratio - 1.0) * 100.0

    delta_r1 = tdpk_stats["r1"] - hover_stats["r1"]
    delta_r2 = tdpk_stats["r2"] - hover_stats["r2"]
    delta_r3 = tdpk_stats["r3"] - hover_stats["r3"]
    delta_r4 = tdpk_stats["r4"] - hover_stats["r4"]
    delta_r5 = tdpk_stats["r5"] - hover_stats["r5"]
    delta_r6 = tdpk_stats["r6"] - hover_stats["r6"]

    print("\n| Component | TDPK (Arrived) | Always Hover (200 steps) | Delta (TDPK - Hover) | Description / Role |")
    print("| :--- | :---: | :---: | :---: | :--- |")
    print(f"| **Steps Taken** | {tdpk_stats['steps']:.1f} steps | {hover_stats['steps']:.1f} steps | {tdpk_stats['steps'] - hover_stats['steps']:+.1f} steps | Faster completion saves time |")
    print(f"| **r1 (Throughput)** | {tdpk_stats['r1']:+.2f} | {hover_stats['r1']:+.2f} | {delta_r1:+.2f} | Throughput gained over trajectory |")
    print(f"| **r2 (Energy Cost)** | {tdpk_stats['r2']:+.2f} | {hover_stats['r2']:+.2f} | {delta_r2:+.2f} | Energy cost of moving vs hovering |")
    print(f"| **r3 (Terminal)** | {tdpk_stats['r3']:+.2f} | {hover_stats['r3']:+.2f} | {delta_r3:+.2f} | Arrival bonus (+1) vs non-arrival penalty |")
    print(f"| **r4 (Proximity)** | {tdpk_stats['r4']:+.2f} | {hover_stats['r4']:+.2f} | {delta_r4:+.2f} | Distance closing bonus vs lack penalty |")
    print(f"| **r5 (Acceleration)** | {tdpk_stats['r5']:+.2f} | {hover_stats['r5']:+.2f} | {delta_r5:+.2f} | Acceleration violation penalties |")
    print(f"| **r6 (Height)** | {tdpk_stats['r6']:+.2f} | {hover_stats['r6']:+.2f} | {delta_r6:+.2f} | Altitude boundary violations |")
    print(f"| **Total Reward** | **{tdpk_stats['total']:+.2f}** | **{hover_stats['total']:+.2f}** | **{delta_total:+.2f}** | **Net Advantage: {ratio:.2f}x ({pct_gain:+.1f}%)** |")

    # Interpretation
    print("\n" + "=" * 80)
    print("DETAILED INTERPRETATION:")
    print("=" * 80)
    print(f"1. Energy Eaten Fraction:")
    print(f"   - Under TDPK, energy cost (r2={tdpk_stats['r2']:.2f}) eats {tdpk_stats['energy_pct_r1']:.1f}% of throughput (r1={tdpk_stats['r1']:.2f}).")
    print(f"   - Under Hover, energy cost (r2={hover_stats['r2']:.2f}) eats {hover_stats['energy_pct_r1']:.1f}% of throughput (r1={hover_stats['r1']:.2f}).")
    print(f"\n2. Source of TDPK's Net Advantage (+{delta_total:.2f}):")
    print(f"   - r1 Throughput: {delta_r1:+.2f} (TDPK covers full field vs hover in corner)")
    print(f"   - r2 Energy:     {delta_r2:+.2f} (TDPK spends energy for fewer steps: {tdpk_stats['steps']:.1f}s vs {hover_stats['steps']:.1f}s)")
    print(f"   - r3 Terminal:   {delta_r3:+.2f} (TDPK avoids non-arrival penalty)")
    print(f"   - r4 Proximity:  {delta_r4:+.2f} (TDPK earns d_near bonuses)")
    print(f"\n3. Margin Analysis:")
    if ratio >= 2.0:
        print(f"   - The advantage is LARGE and robust ({ratio:.2f}x, +{pct_gain:.1f}%).")
        print("   - Completing a full journey is substantially rewarded over hovering.")
        print("   - This suggests the reward function fundamentally favors arrival over hovering when evaluated globally.")
    elif ratio >= 1.3:
        print(f"   - The advantage is MODERATE ({ratio:.2f}x, +{pct_gain:.1f}%).")
        print("   - TDPK beats hover by a clear margin, but hovering still captures a large baseline return.")
    else:
        print(f"   - The advantage is NARROW ({ratio:.2f}x, under +30%).")
        print("   - Hovering captures almost as much reward as full journey, making navigation a weak learning signal.")


if __name__ == "__main__":
    main()
