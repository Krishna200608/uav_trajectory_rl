"""
Trend evaluation script across all checkpoints in checkpoints/diag_actionscale_2500.

Evaluates every saved checkpoint (every 250 episodes) across 10 evaluation seeds (0-9)
to track whether the fraction of moving seeds trends upward over training.
"""

import glob
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from uav_trajectory_rl.config import Q_START
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import unnormalize_action
from uav_trajectory_rl.td3_networks import Actor


def evaluate_checkpoint(ckpt_path: str, num_seeds: int = 10) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor = Actor(state_dim=26, action_dim=3, max_action=1.0)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    records = []
    for seed in range(num_seeds):
        env = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(seed))
        state = env.reset()
        done = False
        ep_reward = 0.0
        max_dist_from_start = 0.0
        q_start = env.q_start.copy()

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_norm = actor(state_t).squeeze(0).numpy()
            v_phys, lam_phys, rho_phys = unnormalize_action(action_norm)

            next_state, reward, done, info = env.step((v_phys, lam_phys, rho_phys))
            cur_dist = float(np.linalg.norm(env.uav_pos - q_start))
            if cur_dist > max_dist_from_start:
                max_dist_from_start = cur_dist

            state = next_state
            ep_reward += reward

        final_disp = float(np.linalg.norm(env.uav_pos - q_start))
        records.append({
            "seed": seed,
            "final_disp": final_disp,
            "max_dist": max_dist_from_start,
            "exceeds_50m": max_dist_from_start >= 50.0,
            "reward": ep_reward,
        })

    mean_max_disp = float(np.mean([r["max_dist"] for r in records]))
    frac_exceeding = float(np.mean([1.0 if r["exceeds_50m"] else 0.0 for r in records]))
    mean_reward = float(np.mean([r["reward"] for r in records]))

    return {
        "mean_max_disp": mean_max_disp,
        "frac_exceeding": frac_exceeding,
        "mean_reward": mean_reward,
        "records": records,
    }


def find_checkpoints(checkpoint_dir: str = "checkpoints/diag_actionscale_2500") -> List[tuple[int, str]]:
    pattern = os.path.join(checkpoint_dir, "td3_agent_ep*.pt")
    files = glob.glob(pattern)
    ckpts = []
    for f in files:
        m = re.search(r"td3_agent_ep(\d+)\.pt", f)
        if m:
            ckpts.append((int(m.group(1)), f))
    ckpts.sort(key=lambda x: x[0])

    final_f = os.path.join(checkpoint_dir, "td3_agent_final.pt")
    if os.path.isfile(final_f):
        # Determine highest episode
        last_ep = ckpts[-1][0] if ckpts else 2500
        ckpts.append((last_ep, final_f))

    # Remove duplicate final if already present
    seen = set()
    unique_ckpts = []
    for ep, f in ckpts:
        if f not in seen:
            seen.add(f)
            unique_ckpts.append((ep, f))
    return unique_ckpts


def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/diag_actionscale_2500"
    ckpts = find_checkpoints(checkpoint_dir)

    print("=" * 85)
    print(f"TREND EVALUATION: DISPLACEMENT & MOVEMENT FRACTION ({len(ckpts)} Checkpoints)")
    print(f"Directory: {checkpoint_dir}")
    print("=" * 85)

    if not ckpts:
        print("No checkpoints found.")
        return

    summary_table = []
    for ep, path in ckpts:
        tag = f"ep{ep}" if "final" not in path else f"ep{ep} (final)"
        print(f"\nEvaluating {tag}: {path} ...", flush=True)
        res = evaluate_checkpoint(path, num_seeds=10)
        summary_table.append({
            "tag": tag,
            "ep": ep,
            "mean_max_disp": res["mean_max_disp"],
            "frac_exceeding": res["frac_exceeding"],
            "mean_reward": res["mean_reward"],
            "records": res["records"],
        })

    print("\n" + "=" * 85)
    print("SUMMARY TREND TABLE ACROSS CHECKPOINTS (10 Evaluation Seeds: 0-9)")
    print("=" * 85)
    print(f"{'Episode / Checkpoint':<25} | {'Mean Max Disp (m)':<20} | {'Frac Exceeding 50m':<22} | {'Mean Reward':<14}")
    print("-" * 85)
    for row in summary_table:
        print(f"{row['tag']:<25} | {row['mean_max_disp']:<20.1f} | {row['frac_exceeding'] * 100:<20.1f}% | {row['mean_reward']:<14.2f}")

    print("\n" + "=" * 85)
    print("SEED-BY-SEED BREAKDOWN (Max Distance from Start in meters)")
    print("=" * 85)
    header = f"{'Checkpoint':<15} | " + " | ".join([f"S{s}" for s in range(10)])
    print(header)
    print("-" * len(header))
    for row in summary_table:
        dists = [f"{r['max_dist']:>4.0f}" for r in row["records"]]
        print(f"{row['tag']:<15} | " + " | ".join(dists))


if __name__ == "__main__":
    main()
