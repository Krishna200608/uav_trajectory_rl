"""
Diagnostic Evaluation Suite for Dueling DQL Baseline (M11).

Evaluates checkpoints over 20 deterministic seeds (seeds 0-19, epsilon=0.0):
  1. Behavioral Rollout Metrics:
     - Mean max displacement, median displacement, fraction > 50m.
     - Destination arrival rate.
     - Mean cumulative episode reward.
  2. Direct Q-Value Inspection at Q_START = (0, 0, 50):
     - Chosen greedy action and physical parameters (v, lam, rho).
     - Q-value of Goal action vs. Hover vs. Wall vs. Chosen action.
     - Q-value spread between Goal and Wall.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from uav_trajectory_rl.baselines.dueling_dql import (
    DuelingDQLAgent,
    discrete_action_to_physical,
    physical_to_nearest_discrete_idx,
)
from uav_trajectory_rl.config import NUM_DISCRETE_ACTIONS
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def rollout_eval(agent: DuelingDQLAgent, seed: int, k: int = 10, max_steps: int = 200) -> tuple[float, float, bool]:
    """Execute a single deterministic evaluation episode (epsilon=0.0)."""
    env = UAVTrajectoryEnv(k=k, rng=np.random.default_rng(seed))
    state = env.reset()
    start_pos = env.uav_pos.copy()
    max_dist = 0.0
    ep_reward = 0.0
    arrived = False

    for _ in range(max_steps):
        action_idx = agent.select_action(state, epsilon=0.0)
        action_phys = discrete_action_to_physical(action_idx)
        state, r, done, info = env.step(action_phys)

        dist = float(np.linalg.norm(env.uav_pos - start_pos))
        if dist > max_dist:
            max_dist = dist
        ep_reward += r

        if info.get("arrived", False):
            arrived = True
        if done:
            break

    return max_dist, ep_reward, arrived


def evaluate_checkpoints(checkpoint_dir: str = "checkpoints/dueling_dql_diag", num_seeds: int = 20):
    ckpt_dir = Path(checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("dueling_dql_ep*.pt"), key=lambda p: int(p.stem.replace("dueling_dql_ep", "")))
    final_ckpt = ckpt_dir / "dueling_dql_final.pt"
    if final_ckpt.exists() and final_ckpt not in ckpts:
        ckpts.append(final_ckpt)

    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir.resolve()}")
        return

    print("=" * 95)
    print(f"DUELING DQL BEHAVIORAL EVALUATION ({num_seeds} Deterministic Seeds, epsilon=0.0)")
    print("=" * 95)
    print(f"{'Checkpoint':<24} | {'MeanMaxDisp':>11} | {'MedianDisp':>11} | {'Frac>50m':>9} | {'ArrivalRate':>11} | {'MeanReward':>10}")
    print("-" * 95)

    env0 = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0))
    s0 = env0.reset()
    state_dim = env0.state_dim

    # Reference action indices
    idx_goal = physical_to_nearest_discrete_idx(20.0, 0.5 * math.pi, 0.25 * math.pi)     # Goal direction
    idx_hover = physical_to_nearest_discrete_idx(0.0, 0.5 * math.pi, 0.0)                # Hover
    idx_wall = physical_to_nearest_discrete_idx(20.0, 0.5 * math.pi, -0.75 * math.pi)   # Into west wall

    results = []

    for ckpt_path in ckpts:
        agent = DuelingDQLAgent(state_dim=state_dim, num_actions=NUM_DISCRETE_ACTIONS)
        agent.load(ckpt_path)

        dists, rewards, arrivals = [], [], []
        for seed in range(num_seeds):
            d, r, a = rollout_eval(agent, seed=seed, k=10)
            dists.append(d)
            rewards.append(r)
            arrivals.append(a)

        dists_arr = np.array(dists)
        arr_rate = float(np.mean(arrivals) * 100.0)
        mean_disp = float(dists_arr.mean())
        med_disp = float(np.median(dists_arr))
        frac_50 = float((dists_arr > 50.0).mean() * 100.0)
        mean_rew = float(np.mean(rewards))

        print(f"{ckpt_path.name:<24} | {mean_disp:10.1f}m | {med_disp:10.1f}m | {frac_50:8.1f}% | {arr_rate:10.1f}% | {mean_rew:10.2f}")

        # Direct Q-value inspection at Q_START
        with torch.no_grad():
            s_t = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
            q_all = agent.q_net(s_t)[0]
            chosen_idx = int(torch.argmax(q_all).item())
            q_goal = float(q_all[idx_goal].item())
            q_hover = float(q_all[idx_hover].item())
            q_wall = float(q_all[idx_wall].item())
            q_chosen = float(q_all[chosen_idx].item())

        chosen_phys = discrete_action_to_physical(chosen_idx)
        spread = abs(q_goal - q_wall) / abs(q_goal) * 100.0 if q_goal != 0.0 else float("nan")

        results.append({
            "ckpt": ckpt_path.name,
            "mean_disp": mean_disp,
            "med_disp": med_disp,
            "frac_50": frac_50,
            "arr_rate": arr_rate,
            "mean_rew": mean_rew,
            "chosen_phys": chosen_phys,
            "q_goal": q_goal,
            "q_hover": q_hover,
            "q_wall": q_wall,
            "q_chosen": q_chosen,
            "spread": spread,
        })

    print("\n" + "=" * 115)
    print("DIRECT Q-VALUE INSPECTION AT Q_START = (0, 0, 50)")
    print("=" * 115)
    print(f"{'Checkpoint':<24} | {'Chosen Action (phys)':<35} | {'Q(Goal)':>9} | {'Q(Hov)':>9} | {'Q(Wall)':>9} | {'Q(Chosen)':>9} | {'Spread A-C':>10}")
    print("-" * 115)
    for r in results:
        v, lam, rho = r["chosen_phys"]
        phys_str = f"v={v:.1f}, lam={math.degrees(lam):.0f}deg, rho={math.degrees(rho):.0f}deg"
        print(f"{r['ckpt']:<24} | {phys_str:<35} | {r['q_goal']:9.2f} | {r['q_hover']:9.2f} | {r['q_wall']:9.2f} | {r['q_chosen']:9.2f} | {r['spread']:9.2f}%")


if __name__ == "__main__":
    evaluate_checkpoints()
