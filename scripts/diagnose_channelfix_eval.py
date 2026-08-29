"""
Evaluate Checkpoints from 800-Episode Channel Fix Diagnostic Run.

Evaluates td3_agent_ep200.pt, td3_agent_ep400.pt, td3_agent_ep600.pt, and td3_agent_final.pt
across 20 seeds (0-19) with deterministic actor.
Reports:
    - Mean Max Displacement (m)
    - Median Max Displacement (m)
    - Fraction > 50m (%)
    - Arrival Rate (%)
    - Mean Cumulative Reward
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from uav_trajectory_rl.config import Q_START, V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import unnormalize_action
from uav_trajectory_rl.td3_networks import Actor


def evaluate_checkpoint_deterministic(ckpt_path: Path, seeds: List[int], k: int = 10) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor = Actor(state_dim=2 * k + 6, action_dim=3, max_action=1.0)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    max_displacements = []
    rewards = []
    arrivals = []

    for seed in seeds:
        env = UAVTrajectoryEnv(k=k, rng=np.random.default_rng(seed))
        state = env.reset()
        done = False
        ep_reward = 0.0
        max_disp = 0.0
        arrived = False

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_norm = actor(state_t).squeeze(0).numpy()
            v_phys, lam_phys, rho_phys = unnormalize_action(action_norm)

            next_state, reward, done, info = env.step((v_phys, lam_phys, rho_phys))
            cur_disp = float(np.linalg.norm(env.uav_pos - env.q_start))
            if cur_disp > max_disp:
                max_disp = cur_disp

            if info.get("arrived", False):
                arrived = True

            state = next_state
            ep_reward += reward

        max_displacements.append(max_disp)
        rewards.append(ep_reward)
        arrivals.append(arrived)

    mean_max_disp = float(np.mean(max_displacements))
    median_max_disp = float(np.median(max_displacements))
    frac_gt_50m = float(np.mean([1.0 if d >= 50.0 else 0.0 for d in max_displacements])) * 100.0
    arrival_rate = float(np.mean([1.0 if a else 0.0 for a in arrivals])) * 100.0
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    return {
        "checkpoint": ckpt_path.name,
        "mean_max_disp": mean_max_disp,
        "median_max_disp": median_max_disp,
        "frac_gt_50m": frac_gt_50m,
        "arrival_rate": arrival_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 800-ep channel fix diagnostic checkpoints")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/diag_channelfix")
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = [
        ckpt_dir / "td3_agent_ep200.pt",
        ckpt_dir / "td3_agent_ep400.pt",
        ckpt_dir / "td3_agent_ep600.pt",
        ckpt_dir / "td3_agent_final.pt",
    ]

    seeds = list(range(args.seeds))
    results = []

    print(f"Evaluating {len(ckpts)} checkpoints over {len(seeds)} seeds (deterministic evaluation)...")
    for ckpt in ckpts:
        if not ckpt.exists():
            print(f"Warning: {ckpt} not found yet.")
            continue
        res = evaluate_checkpoint_deterministic(ckpt, seeds)
        results.append(res)

    print("\n### 800-Episode Diagnostic: Deterministic Evaluation (20 seeds, k=10)")
    print("| Checkpoint | Mean Max Disp (m) | Median Disp (m) | Frac > 50m (%) | Arrival Rate (%) | Mean Reward |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(
            f"| {r['checkpoint']} | {r['mean_max_disp']:6.1f}m | {r['median_max_disp']:6.1f}m "
            f"| {r['frac_gt_50m']:5.1f}% | **{r['arrival_rate']:5.1f}%** | {r['mean_reward']:+7.2f} |"
        )


if __name__ == "__main__":
    main()
