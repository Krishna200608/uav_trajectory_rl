"""
Reward-baseline diagnostic: compare Hover vs TDPK vs Prior-Knowledge Policy.

Tests whether a policy that actively flies toward the destination outperforms
hovering under the project's reward function, and checks whether the replay buffer
ever receives an arrival example during the prior-knowledge exploration phase.
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

from uav_trajectory_rl.config import (
    ARRIVAL_THRESHOLD_M,
    N_SLOTS,
    Q_END,
    Q_START,
    V_MAX,
)
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.baselines.tdpk import tdpk_action
from uav_trajectory_rl.prior_knowledge_policy import generate_prior_knowledge_action


def run_episode(env: UAVTrajectoryEnv, policy_type: str, rng: np.random.Generator) -> Dict:
    """
    Run one complete episode under a specific heuristic policy.

    policy_type:
        - "hover": v=0.0, lam=pi/2, rho=0.0
        - "tdpk": direct-to-destination 3D vector with randomized speed
        - "pk": prior-knowledge policy (lam=LAMBDA_PK, random azimuth cone)
    """
    state = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0
    arrived = False

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
        elif policy_type == "pk":
            action = generate_prior_knowledge_action(rng=rng)
        else:
            raise ValueError(f"Unknown policy: {policy_type}")

        next_state, reward, done, info = env.step(action)
        ep_reward += reward
        steps += 1

        dist_to_end = float(np.linalg.norm(env.uav_pos - env.q_end))
        if dist_to_end <= env.arrival_threshold:
            arrived = True

    final_dist = float(np.linalg.norm(env.uav_pos - env.q_end))
    final_disp = float(np.linalg.norm(env.uav_pos - env.q_start))

    return {
        "reward": ep_reward,
        "arrived": arrived,
        "steps": steps,
        "final_dist": final_dist,
        "final_disp": final_disp,
        "final_pos": env.uav_pos.round(1).tolist(),
    }


def evaluate_policy_across_seeds(policy_type: str, seeds: List[int], k: int = 10) -> List[Dict]:
    results = []
    for seed in seeds:
        env = UAVTrajectoryEnv(k=k, rng=np.random.default_rng(seed))
        res = run_episode(env, policy_type, rng=np.random.default_rng(seed + 10000))
        res["seed"] = seed
        results.append(res)
    return results


def main():
    print("=" * 80)
    print("STEP 1: THREE-WAY REWARD COMPARISON (20 SEEDS, k=10)")
    print("Policies: 1) Always Hover | 2) TDPK (Direct Flight) | 3) Prior-Knowledge Heuristic")
    print("=" * 80)

    seeds_20 = list(range(20))
    res_hover = evaluate_policy_across_seeds("hover", seeds_20)
    res_tdpk = evaluate_policy_across_seeds("tdpk", seeds_20)
    res_pk = evaluate_policy_across_seeds("pk", seeds_20)

    print(f"\n{'Seed':<5} | {'Hover Reward':<14} | {'TDPK Reward':<14} | {'TDPK Arrived':<13} | {'PK Reward':<14} | {'PK Arrived':<11}")
    print("-" * 85)
    for i in range(20):
        h = res_hover[i]
        t = res_tdpk[i]
        p = res_pk[i]
        t_arr = "YES" if t["arrived"] else f"NO ({t['final_dist']:.0f}m)"
        p_arr = "YES" if p["arrived"] else f"NO ({p['final_dist']:.0f}m)"
        print(f"{i:<5} | {h['reward']:<14.2f} | {t['reward']:<14.2f} | {t_arr:<13} | {p['reward']:<14.2f} | {p_arr:<11}")

    # Summary Statistics
    rewards_hover = [r["reward"] for r in res_hover]
    rewards_tdpk = [r["reward"] for r in res_tdpk]
    rewards_pk = [r["reward"] for r in res_pk]

    arr_hover = sum(1 for r in res_hover if r["arrived"]) / len(res_hover)
    arr_tdpk = sum(1 for r in res_tdpk if r["arrived"]) / len(res_tdpk)
    arr_pk = sum(1 for r in res_pk if r["arrived"]) / len(res_pk)

    steps_tdpk = [r["steps"] for r in res_tdpk if r["arrived"]]
    avg_steps_tdpk = np.mean(steps_tdpk) if steps_tdpk else float("nan")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON (20 Seeds)")
    print("=" * 80)
    print(f"{'Policy':<25} | {'Mean Reward':<14} | {'Std Reward':<12} | {'Arrival Rate':<14} | {'Avg Steps (Arrived)':<20}")
    print("-" * 92)
    print(f"{'Always Hover':<25} | {np.mean(rewards_hover):<14.2f} | {np.std(rewards_hover):<12.2f} | {arr_hover * 100:<13.1f}% | {'N/A':<20}")
    print(f"{'TDPK (Direct Flight)':<25} | {np.mean(rewards_tdpk):<14.2f} | {np.std(rewards_tdpk):<12.2f} | {arr_tdpk * 100:<13.1f}% | {avg_steps_tdpk:<20.1f}")
    print(f"{'Prior-Knowledge Policy':<25} | {np.mean(rewards_pk):<14.2f} | {np.std(rewards_pk):<12.2f} | {arr_pk * 100:<13.1f}% | {'N/A':<20}")

    # ==========================================================================
    # STEP 2: ARRIVAL RATE DURING ACTUAL PRIOR-KNOWLEDGE PHASE (150 EPISODES)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: SIMULATED PRIOR-KNOWLEDGE (PK) PHASE ARRIVAL AUDIT (150 EPISODES)")
    print("Evaluating whether the replay buffer ever experiences an arrival event (arrived=True)")
    print("using generate_prior_knowledge_action across 150 independent episodes.")
    print("=" * 80)

    seeds_150 = list(range(150))
    res_pk_150 = evaluate_policy_across_seeds("pk", seeds_150)

    pk_arrivals = [r for r in res_pk_150 if r["arrived"]]
    pk_arrival_count = len(pk_arrivals)
    pk_arrival_rate = pk_arrival_count / 150.0
    pk_final_dists = [r["final_dist"] for r in res_pk_150]

    print(f"Total PK Episodes Tested:       150")
    print(f"Total Successful Arrivals:      {pk_arrival_count} / 150 ({pk_arrival_rate * 100:.2f}%)")
    if pk_arrival_count > 0:
        steps_pk_arr = [r["steps"] for r in pk_arrivals]
        print(f"Steps Taken on Arrival:         Min={min(steps_pk_arr)}, Max={max(steps_pk_arr)}, Mean={np.mean(steps_pk_arr):.1f}")
    else:
        print(f"Steps Taken on Arrival:         None arrived.")
    print(f"Distance to Goal at Term:       Min={min(pk_final_dists):.2f}m, Max={max(pk_final_dists):.2f}m, Mean={np.mean(pk_final_dists):.2f}m")

    # ==========================================================================
    # STEP 3: INTERPRETATION & RECOMMENDATION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: INTERPRETATION & ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    diff_reward = np.mean(rewards_tdpk) - np.mean(rewards_hover)
    print(f"1. Does TDPK outperform 'Always Hover'?")
    if diff_reward > 0:
        print(f"   YES. TDPK mean reward ({np.mean(rewards_tdpk):.2f}) exceeds Hover ({np.mean(rewards_hover):.2f}) by +{diff_reward:.2f} points.")
        print(f"   Significance: The reward function ITSELF does NOT intrinsically prefer hovering over valid goal arrival.")
    else:
        print(f"   NO. TDPK mean reward ({np.mean(rewards_tdpk):.2f}) is WORSE than Hover ({np.mean(rewards_hover):.2f}) by {diff_reward:.2f} points.")
        print(f"   Significance: A deeper reward balance defect exists where even a perfect direct flight earns less than standing still.")

    print(f"\n2. What is TDPK's arrival rate across 20 seeds?")
    print(f"   TDPK Arrival Rate: {arr_tdpk * 100:.1f}% ({sum(1 for r in res_tdpk if r['arrived'])} / 20).")

    print(f"\n3. What is the Prior-Knowledge (PK) policy arrival rate over 150 episodes?")
    print(f"   PK Arrival Rate: {pk_arrival_rate * 100:.2f}% ({pk_arrival_count} / 150).")
    print(f"   Minimum distance achieved by PK policy: {min(pk_final_dists):.2f}m (Threshold is {ARRIVAL_THRESHOLD_M}m).")
    print(f"   Structural insight: Equation (30) defines LAMBDA_PK = pi/2 (strictly horizontal flight).")
    print(f"   Since the UAV starts at z = 50m and destination is at z = 100m, a strictly horizontal")
    print(f"   flight policy CANNOT physically reach altitude 100m! The vertical gap is always |100 - 50| = 50m,")
    print(f"   which is 10x larger than ARRIVAL_THRESHOLD_M = 5.0m.")

    print(f"\n4. Remediation Assessment: (a) Extend R_RAND vs (b) Reward / Exploration Adjustment:")
    if pk_arrival_count == 0:
        print(f"   CRITICAL DIAGNOSTIC VERDICT: Direction (a) alone (extending R_RAND) CANNOT solve the problem")
        print(f"   under the paper's horizontal LAMBDA_PK policy because the PK exploration policy has a ZERO PERCENT")
        print(f"   probability of ever reaching the destination (it never climbs to z=100m).")
        print(f"   The replay buffer will NEVER contain a successful arrival transition during the PK phase regardless")
        print(f"   of whether R_RAND is 20,000, 100,000, or 1,000,000 steps.")


if __name__ == "__main__":
    main()
