"""
Channel Calibration Diagnostic: Sweep FC_HZ and N0_DBM_HZ over TDPK vs Hover.

Sweeps:
    FC_HZ: [2.0e9, 2.4e9, 5.0e9, 28.0e9]
    N0_DBM_HZ: [-174, -169, -164]

For each of the 12 combinations, evaluates TDPK and Always-Hover across 30 seeds (k=10)
to find the throughput-to-energy balance that yields a physically plausible, robust
learning margin over hovering.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import uav_trajectory_rl.channel_model as cm
import uav_trajectory_rl.config as cfg
from uav_trajectory_rl.baselines.tdpk import tdpk_action
from uav_trajectory_rl.channel_model import transmission_rate
import uav_trajectory_rl.mdp_environment as mdp
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def run_episode_breakdown(
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

    return {
        "arrived": arrived,
        "steps": step_count,
        "r1": total_r1,
        "r2": total_r2,
        "r3": total_r3,
        "r4": total_r4,
        "r5": total_r5,
        "r6": total_r6,
        "total_reward": total_reward,
    }


def evaluate_sweep(
    fc_list: List[float],
    n0_list: List[float],
    num_seeds: int = 30,
    k: int = 10,
) -> List[Dict]:
    seeds = list(range(num_seeds))
    results = []

    # Mid-range representative point (300m 3D distance, 100m altitude, 282.84m horizontal)
    uav_pos_ref = [0.0, 0.0, 100.0]
    user_pos_ref = [282.84, 0.0, 0.0]

    orig_ttr = mdp.total_transmission_rate

    try:
        for fc in fc_list:
            for n0 in n0_list:
                # Monkeypatch total_transmission_rate in mdp_environment with explicit parameters
                mdp.total_transmission_rate = (
                    lambda uav_pos, user_pos, _fc=fc, _n0=n0: orig_ttr(
                        uav_pos, user_pos, fc_hz=_fc, n0_dbm_hz=_n0
                    )
                )

                # Calculate reference link rate at 300m
                rate_ref_bps = transmission_rate(
                    uav_pos=uav_pos_ref,
                    user_pos=user_pos_ref,
                    num_users_k=k,
                    fc_hz=fc,
                    n0_dbm_hz=n0,
                )

                # Evaluate TDPK
                tdpk_runs = []
                for seed in seeds:
                    env_rng = np.random.default_rng(seed)
                    action_rng = np.random.default_rng(seed + 10000)
                    env = UAVTrajectoryEnv(k=k, rng=env_rng)
                    rec = run_episode_breakdown(env, "tdpk", action_rng)
                    tdpk_runs.append(rec)

                arrived_tdpk = [r for r in tdpk_runs if r["arrived"]]
                tdpk_mean_r1 = float(np.mean([r["r1"] for r in arrived_tdpk]))
                tdpk_mean_r2 = float(np.mean([r["r2"] for r in arrived_tdpk]))
                tdpk_mean_total = float(np.mean([r["total_reward"] for r in arrived_tdpk]))
                tdpk_mean_steps = float(np.mean([r["steps"] for r in arrived_tdpk]))

                # Evaluate Hover
                hover_runs = []
                for seed in seeds:
                    env_rng = np.random.default_rng(seed)
                    action_rng = np.random.default_rng(seed + 10000)
                    env = UAVTrajectoryEnv(k=k, rng=env_rng)
                    rec = run_episode_breakdown(env, "hover", action_rng)
                    hover_runs.append(rec)

                hover_mean_r1 = float(np.mean([r["r1"] for r in hover_runs]))
                hover_mean_r2 = float(np.mean([r["r2"] for r in hover_runs]))
                hover_mean_total = float(np.mean([r["total_reward"] for r in hover_runs]))

                ratio = tdpk_mean_total / hover_mean_total if hover_mean_total != 0 else float("inf")
                delta_total = tdpk_mean_total - hover_mean_total

                results.append({
                    "fc_hz": fc,
                    "fc_ghz": fc / 1e9,
                    "n0_dbm_hz": n0,
                    "rate_ref_mbps": rate_ref_bps / 1e6,
                    "rate_ref_bps": rate_ref_bps,
                    "tdpk_r1": tdpk_mean_r1,
                    "tdpk_r2": tdpk_mean_r2,
                    "tdpk_total": tdpk_mean_total,
                    "tdpk_steps": tdpk_mean_steps,
                    "hover_r1": hover_mean_r1,
                    "hover_r2": hover_mean_r2,
                    "hover_total": hover_mean_total,
                    "ratio": ratio,
                    "delta": delta_total,
                })
    finally:
        mdp.total_transmission_rate = orig_ttr

    return results


def print_table(results: List[Dict]):
    print("\n### Channel Calibration Sweep (12 Combinations, 30 Seeds, k=10)")
    print("| FC (GHz) | N0 (dBm/Hz) | Ref Rate @300m | TDPK r1 | TDPK Total | Hover r1 | Hover Total | Delta (TDPK-Hover) | TDPK/Hover Margin |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        ratio_str = f"{r['ratio']:.2f}x" if not math.isinf(r["ratio"]) else "inf"
        print(
            f"| {r['fc_ghz']:4.1f} GHz | {int(r['n0_dbm_hz']):4d} | {r['rate_ref_mbps']:6.2f} Mbps "
            f"| {r['tdpk_r1']:+8.1f} | {r['tdpk_total']:+8.1f} "
            f"| {r['hover_r1']:+8.1f} | {r['hover_total']:+8.1f} "
            f"| {r['delta']:+8.1f} | **{ratio_str}** |"
        )


def main():
    fcs = [2.0e9, 2.4e9, 5.0e9, 28.0e9]
    n0s = [-174.0, -169.0, -164.0]

    print("Running 12-combination channel calibration sweep across 30 seeds...")
    results = evaluate_sweep(fcs, n0s, num_seeds=30, k=10)
    print_table(results)


if __name__ == "__main__":
    main()
