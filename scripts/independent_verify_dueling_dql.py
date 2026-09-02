"""
INDEPENDENT verification (Claude, fresh session) of the Dueling DQL 30-seed
deterministic evaluation numbers reported in docs/PKTD3-TD_Tracker.md:
    mean min-dist-to-Q_END = 310.6 m (median 300.7 m, range 162.9-541.2 m)
    mean boundary cancellation in last 50 steps = 17.7%
    arrival rate = 0.0%
    mean max displacement = 600.2 m (median 602.8 m)

This script is written independently (not copied from scripts/diagnose_dueling_dql.py)
to serve as a genuine cross-check, following this project's established discipline of
never trusting a reported number without re-deriving it from the actual checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch

from uav_trajectory_rl.baselines.dueling_dql import DuelingDQLAgent, discrete_action_to_physical
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.config import Q_END

CKPT_PATH = "checkpoints/dueling_dql_run1/dueling_dql_final.pt"
NUM_SEEDS = 30
K_USERS = 10
MAX_STEPS = 200

def run_episode(agent: DuelingDQLAgent, seed: int):
    env = UAVTrajectoryEnv(k=K_USERS, rng=np.random.default_rng(seed))
    state = env.reset()
    start_pos = env.uav_pos.copy()
    q_end = np.array(Q_END)

    max_disp = 0.0
    min_dist_to_end = float(np.linalg.norm(env.uav_pos - q_end))
    cancels_last50 = 0
    arrived = False

    for t in range(MAX_STEPS):
        action_idx = agent.select_action(state, epsilon=0.0)
        v, lam, rho = discrete_action_to_physical(action_idx)
        state, r, done, info = env.step((v, lam, rho))

        disp = float(np.linalg.norm(env.uav_pos - start_pos))
        max_disp = max(max_disp, disp)

        dist_to_end = float(np.linalg.norm(env.uav_pos - q_end))
        min_dist_to_end = min(min_dist_to_end, dist_to_end)

        if t >= MAX_STEPS - 50:
            if info.get("position_cancelled", False):
                cancels_last50 += 1

        if info.get("arrived", False):
            arrived = True
        if done:
            break

    final_dist_to_end = float(np.linalg.norm(env.uav_pos - q_end))
    cancel_frac_last50 = cancels_last50 / 50.0
    return {
        "seed": seed,
        "max_disp": max_disp,
        "min_dist_to_end": min_dist_to_end,
        "final_dist_to_end": final_dist_to_end,
        "cancel_frac_last50": cancel_frac_last50,
        "arrived": arrived,
    }


def main():
    # Load agent fresh, purely from the checkpoint on disk.
    probe_env = UAVTrajectoryEnv(k=K_USERS, rng=np.random.default_rng(0))
    probe_env.reset()
    state_dim = probe_env.state_dim

    agent = DuelingDQLAgent(state_dim=state_dim, seed=0)
    agent.load(CKPT_PATH)
    agent.q_net.eval()

    results = [run_episode(agent, seed) for seed in range(NUM_SEEDS)]

    max_disps = np.array([r["max_disp"] for r in results])
    min_dists = np.array([r["min_dist_to_end"] for r in results])
    final_dists = np.array([r["final_dist_to_end"] for r in results])
    cancel_fracs = np.array([r["cancel_frac_last50"] for r in results])
    arrivals = np.array([r["arrived"] for r in results])

    print("=" * 80)
    print(f"INDEPENDENT RE-VERIFICATION: Dueling DQL ({CKPT_PATH}), {NUM_SEEDS} seeds")
    print("=" * 80)
    print(f"Mean max displacement:        {max_disps.mean():.1f} m  (median {np.median(max_disps):.1f} m)")
    print(f"Mean min dist to Q_END:       {min_dists.mean():.1f} m  (median {np.median(min_dists):.1f} m, "
          f"range {min_dists.min():.1f}-{min_dists.max():.1f} m)")
    print(f"Mean final dist to Q_END:     {final_dists.mean():.1f} m  (median {np.median(final_dists):.1f} m)")
    print(f"Mean cancellation (last 50):  {cancel_fracs.mean()*100:.1f} %")
    print(f"Arrival rate:                 {arrivals.mean()*100:.1f} % ({arrivals.sum()}/{NUM_SEEDS})")
    print()
    print("Reported in tracker: min_dist mean=310.6m median=300.7m range=162.9-541.2m, "
          "canc_last50=17.7%, arrival=0.0%, max_disp mean=600.2m median=602.8m")
    print()
    print("Per-seed min-dist-to-end (sorted):")
    for r in sorted(results, key=lambda x: x["min_dist_to_end"]):
        print(f"  seed {r['seed']:>2}: min_dist={r['min_dist_to_end']:>6.1f}m  "
              f"canc_last50={r['cancel_frac_last50']*100:>5.1f}%  arrived={r['arrived']}")


if __name__ == "__main__":
    main()
