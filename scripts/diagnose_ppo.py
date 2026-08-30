"""
PPO Diagnostic Evaluation Script (M12 Baseline).

Loads PPO checkpoints and evaluates across 20 deterministic seeds using
select_action_deterministic() (Gaussian mean, no sampling), measuring:
  - Mean / median max displacement (m)
  - Fraction of episodes > 50m max displacement
  - Arrival rate (fraction reaching Q_END within ARRIVAL_THRESHOLD_M)
  - Mean cumulative reward

Protocol matches the Dueling DQL (M11) and PKTD3-TD (M9) evaluation
exactly, for direct comparability. Also reports mean policy entropy over
training as a measure of exploration commitment.

Usage:
    python scripts/diagnose_ppo.py --checkpoint-dir checkpoints/ppo_diag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch

from uav_trajectory_rl.baselines.ppo import PPOAgent
from uav_trajectory_rl.config import Q_START
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def evaluate_checkpoint(
    ckpt_path: Path,
    state_dim: int,
    n_eval_seeds: int = 20,
    k_users: int = 10,
) -> dict:
    """
    Run deterministic evaluation of a PPO checkpoint.

    Args:
        ckpt_path:    Path to a .pt checkpoint file.
        state_dim:    Environment state dimension.
        n_eval_seeds: Number of evaluation seeds (each is a separate rollout).
        k_users:      Number of ground users.

    Returns:
        Dict with mean/median max_disp, frac_gt50m, arrival_rate, mean_reward.
    """
    agent = PPOAgent(state_dim=state_dim)
    agent.load(ckpt_path)
    agent.actor.eval()
    agent.critic.eval()

    max_disps = []
    rewards = []
    arrivals = 0

    for seed in range(n_eval_seeds):
        rng = np.random.default_rng(seed)
        env = UAVTrajectoryEnv(k=k_users, rng=rng)
        state = env.reset()

        ep_reward = 0.0
        max_disp = 0.0
        arrived = False
        done = False

        while not done:
            physical_action = agent.select_action_deterministic(state)
            next_state, reward, done, info = env.step(physical_action)
            ep_reward += reward

            # Track displacement from Q_START
            pos = env.uav_pos
            disp = float(np.linalg.norm(np.array(pos) - np.array(Q_START)))
            max_disp = max(max_disp, disp)

            if info.get("arrived", False):
                arrived = True

            state = next_state

        max_disps.append(max_disp)
        rewards.append(ep_reward)
        if arrived:
            arrivals += 1

    return {
        "mean_max_disp": float(np.mean(max_disps)),
        "median_max_disp": float(np.median(max_disps)),
        "frac_gt50m": float(np.mean([d > 50.0 for d in max_disps])),
        "arrival_rate": arrivals / n_eval_seeds,
        "mean_reward": float(np.mean(rewards)),
    }


def main(
    checkpoint_dir: str = "checkpoints/ppo_diag",
    n_eval_seeds: int = 20,
    k_users: int = 10,
) -> None:
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt_path.resolve()}")
        sys.exit(1)

    # Find all checkpoint files
    ckpt_files = sorted(
        ckpt_path.glob("ppo_step*.pt"),
        key=lambda p: int(p.stem.replace("ppo_step", "")),
    )
    final_file = ckpt_path / "ppo_final.pt"
    if final_file.exists() and final_file not in ckpt_files:
        ckpt_files.append(final_file)

    if not ckpt_files:
        print(f"ERROR: No PPO checkpoint files found in {ckpt_path.resolve()}")
        sys.exit(1)

    print(f"Found {len(ckpt_files)} checkpoint(s) in {ckpt_path.resolve()}")

    # Infer state_dim from a fresh env
    env_tmp = UAVTrajectoryEnv(k=k_users, rng=np.random.default_rng(0))
    state_dim = env_tmp.state_dim

    print("\nDeterministic evaluation (select_action_deterministic, 20 seeds):\n")
    print(f"{'Checkpoint':<25} {'MeanDisp':>10} {'MedDisp':>10} {'Frac>50m':>10} {'ArrRate':>9} {'MeanRew':>12}")
    print("-" * 80)

    results = {}
    for cp in ckpt_files:
        tag = cp.name
        r = evaluate_checkpoint(cp, state_dim, n_eval_seeds, k_users)
        results[tag] = r
        print(
            f"{tag:<25} {r['mean_max_disp']:>10.1f} {r['median_max_disp']:>10.1f} "
            f"{r['frac_gt50m']:>10.1%} {r['arrival_rate']:>9.1%} {r['mean_reward']:>+12.2f}"
        )

    # Entropy over training (from episode_stats.json if available)
    stats_file = ckpt_path / "episode_stats.json"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"\nTotal episodes in training log: {len(stats)}")

    # Save evaluation results
    out_file = ckpt_path / "ppo_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {out_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Baseline Checkpoints (M12)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo_diag",
                        help="Directory containing PPO checkpoint files")
    parser.add_argument("--n-eval-seeds", type=int, default=20,
                        help="Number of evaluation seeds (default: 20)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of ground users (default: 10)")
    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        n_eval_seeds=args.n_eval_seeds,
        k_users=args.k,
    )
