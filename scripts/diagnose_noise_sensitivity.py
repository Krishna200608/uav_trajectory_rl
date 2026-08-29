"""
Noise-sensitivity diagnostic for PKTD3-TD Run 3 checkpoints.

Evaluates trained checkpoints (e.g. td3_agent_ep6000.pt and td3_agent_ep4000.pt)
across a sweep of evaluation noise levels (sigma_eval in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
added to the actor's normalized output before clipping and un-normalizing:
    a_norm = clip(actor(state) + N(0, sigma_eval), -c, c)
    a_phys = unnormalize_action(a_norm, c)

Evaluates 30 seeds (0-29) per noise level to determine:
1. Whether exploration noise was doing all the navigation during training.
2. Whether the deterministic policy (sigma_eval=0.0) failed to consolidate the flight trajectory.
3. Whether the critic's value landscape supports navigation when noise is reintroduced.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from uav_trajectory_rl.config import ACTION_CLIP_C, Q_START, SIGMA3
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import unnormalize_action
from uav_trajectory_rl.td3_networks import Actor


def evaluate_checkpoint_noise_sweep(
    ckpt_path: str,
    sigmas: List[float],
    num_seeds: int = 30,
    k: int = 10,
) -> Dict[float, Dict[str, float]]:
    """
    Run evaluation noise sweep for a single checkpoint across num_seeds.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor = Actor(state_dim=2 * k + 6, action_dim=3, max_action=ACTION_CLIP_C)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    sweep_results = {}

    for sigma_eval in sigmas:
        max_disps = []
        final_disps = []
        exceeds_50m = []
        arrivals = []
        rewards = []

        for seed in range(num_seeds):
            # Use distinct deterministic RNGs so the user mobility for seed i
            # is identical across all sigma_eval values, isolating noise impact.
            env_rng = np.random.default_rng(seed)
            noise_rng = np.random.default_rng(seed + 10000)

            env = UAVTrajectoryEnv(k=k, rng=env_rng)
            state = env.reset()
            q_start = env.q_start.copy()

            done = False
            ep_reward = 0.0
            max_dist = 0.0
            arrived = False

            while not done:
                state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action_norm = actor(state_t).squeeze(0).cpu().numpy()

                if sigma_eval > 0.0:
                    noise = noise_rng.normal(0.0, sigma_eval, size=3)
                    noisy_norm = np.clip(action_norm + noise, -ACTION_CLIP_C, ACTION_CLIP_C)
                else:
                    noisy_norm = action_norm

                action_phys = unnormalize_action(noisy_norm, c=ACTION_CLIP_C)
                next_state, reward, done, info = env.step(action_phys)

                cur_dist = float(np.linalg.norm(env.uav_pos - q_start))
                if cur_dist > max_dist:
                    max_dist = cur_dist

                if info.get("arrived", False):
                    arrived = True

                state = next_state
                ep_reward += reward

            final_disp = float(np.linalg.norm(env.uav_pos - q_start))
            max_disps.append(max_dist)
            final_disps.append(final_disp)
            exceeds_50m.append(1.0 if max_dist >= 50.0 else 0.0)
            arrivals.append(1.0 if arrived else 0.0)
            rewards.append(ep_reward)

        sweep_results[sigma_eval] = {
            "mean_max_disp": float(np.mean(max_disps)),
            "median_max_disp": float(np.median(max_disps)),
            "frac_gt_50m": float(np.mean(exceeds_50m) * 100.0),
            "arrival_rate": float(np.mean(arrivals) * 100.0),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
        }

    return sweep_results


def print_sweep_table(ckpt_name: str, results: Dict[float, Dict[str, float]]):
    print(f"\n### Noise Sensitivity Sweep: {ckpt_name}")
    print("| sigma_eval | Mean Max Disp (m) | Median Disp (m) | Frac > 50m (%) | Arrival Rate (%) | Mean Reward | Std Reward |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for sigma, res in results.items():
        print(
            f"| {sigma:.2f} | {res['mean_max_disp']:.1f}m | {res['median_max_disp']:.1f}m "
            f"| {res['frac_gt_50m']:.1f}% | {res['arrival_rate']:.1f}% "
            f"| {res['mean_reward']:+.2f} | {res['std_reward']:.2f} |"
        )


def analyze_results(
    all_results: Dict[str, Dict[float, Dict[str, float]]],
) -> str:
    lines = []
    lines.append("\n============================================================")
    lines.append("INTERPRETATION & DIAGNOSTIC READOUT")
    lines.append("============================================================")

    for ckpt_name, sweep in all_results.items():
        det = sweep[0.0]
        sig_train = sweep.get(0.1, sweep.get(0.10))
        max_noise = sweep[max(sweep.keys())]

        max_arr = max(res["arrival_rate"] for res in sweep.values())
        sigma_at_max_arr = [s for s, res in sweep.items() if res["arrival_rate"] == max_arr][0]

        lines.append(f"\n--- Analysis for {ckpt_name} ---")
        lines.append(f"  - Deterministic (sigma=0.0): Mean Disp={det['mean_max_disp']:.1f}m, "
                     f"Median={det['median_max_disp']:.1f}m, Frac>50m={det['frac_gt_50m']:.1f}%, "
                     f"Arrival Rate={det['arrival_rate']:.1f}%, Mean Reward={det['mean_reward']:+.2f}")
        if sig_train:
            lines.append(f"  - Training Noise (sigma=0.1): Mean Disp={sig_train['mean_max_disp']:.1f}m, "
                         f"Median={sig_train['median_max_disp']:.1f}m, Frac>50m={sig_train['frac_gt_50m']:.1f}%, "
                         f"Arrival Rate={sig_train['arrival_rate']:.1f}%, Mean Reward={sig_train['mean_reward']:+.2f}")
        lines.append(f"  - Peak Arrival Rate in Sweep: {max_arr:.1f}% at sigma_eval={sigma_at_max_arr:.2f}")

    lines.append("\nSynthesis against Diagnostic Criteria:")
    # Comparison between checkpoints
    ckpt_names = list(all_results.keys())
    if len(ckpt_names) >= 2:
        c1, c2 = ckpt_names[0], ckpt_names[1]
        c1_max_arr = max(res["arrival_rate"] for res in all_results[c1].values())
        c2_max_arr = max(res["arrival_rate"] for res in all_results[c2].values())
        lines.append(f"1. Checkpoint Comparison ({c1} vs {c2}):")
        lines.append(f"   Peak arrival rate: {c1} = {c1_max_arr:.1f}% vs {c2} = {c2_max_arr:.1f}%.")

    # Overall behavior
    global_max_arrival = max(
        max(res["arrival_rate"] for res in sweep.values())
        for sweep in all_results.values()
    )
    if global_max_arrival >= 15.0:
        lines.append("2. Behavior Verdict: NONZERO ARRIVAL UNDER NOISE (>=15%).")
        lines.append("   Noise successfully activates navigation paths toward the destination.")
        lines.append("   This indicates the critic's value gradients reward forward navigation, but")
        lines.append("   the deterministic actor mean has not yet consolidated around that peak.")
    else:
        lines.append(f"2. Behavior Verdict: ARRIVAL REMAINS ZERO/NEAR-ZERO ({global_max_arrival:.1f}% max across all noise levels).")
        lines.append("   Even with exploratory perturbation up to sigma=0.5, the policy does not navigate to Q_END.")
        lines.append("   This confirms a deeper navigation/credit assignment bottleneck rather than pure noise reliance.")

    report_str = "\n".join(lines)
    print(report_str)
    return report_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate noise sensitivity of Run 3 checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/run3/td3_agent_ep6000.pt",
            "checkpoints/run3/td3_agent_ep4000.pt",
        ],
        help="Paths to checkpoint .pt files to evaluate.",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        help="Evaluation noise standard deviations.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of evaluation seeds (default: 30).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of ground users (default: 10).",
    )
    args = parser.parse_args()

    all_results = {}
    for ckpt_path in args.checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nEvaluating {ckpt_name} over {args.seeds} seeds across sigmas: {args.sigmas}...")
        results = evaluate_checkpoint_noise_sweep(
            ckpt_path=ckpt_path,
            sigmas=args.sigmas,
            num_seeds=args.seeds,
            k=args.k,
        )
        all_results[ckpt_name] = results
        print_sweep_table(ckpt_name, results)

    report = analyze_results(all_results)


if __name__ == "__main__":
    main()
