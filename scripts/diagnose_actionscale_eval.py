"""
Diagnostic evaluation script for action-scale fix.

Evaluates all saved checkpoints (ep200, ep400, ep600, ep800/final) from
checkpoints/diag_actionscale_fix:
1. Behavioral rollouts (10 seeds 0-9) measuring displacement, max distance, and reward.
2. Critic sanity check: verifying Q1 across physical and normalized actions.
"""

import math
import os
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
from uav_trajectory_rl.prior_knowledge_policy import normalize_action, unnormalize_action
from uav_trajectory_rl.td3_networks import Actor, TwinCritic


def evaluate_checkpoint_behavior(ckpt_path: str, num_seeds: int = 10) -> List[Dict]:
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
    return records


def evaluate_critic_consistency(ckpt_path: str, test_state: np.ndarray):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    critic = TwinCritic(state_dim=26, action_dim=3)
    critic.load_state_dict(ckpt["critic"])
    critic.eval()

    state_t = torch.as_tensor(test_state, dtype=torch.float32).unsqueeze(0)

    # Test speeds spanning 0 to V_MAX
    test_speeds = np.linspace(0.0, V_MAX, 9)
    # Sensible flight direction: horizontal flight towards destination (+x, +y) -> lam = pi/2, rho = pi/4
    lam_phys = math.pi / 2.0
    rho_phys = math.pi / 4.0

    results = []
    for v_phys in test_speeds:
        # 1. Expressed via normalize_action
        phys_tuple = (float(v_phys), float(lam_phys), float(rho_phys))
        norm_arr = normalize_action(phys_tuple)
        act_norm_t = torch.as_tensor(norm_arr, dtype=torch.float32).unsqueeze(0)

        # 2. Directly constructed normalized action
        # v_raw = v / V_MAX * 2 - 1; lam_raw = 0.0; rho_raw = 0.25
        v_raw_expected = (v_phys / V_MAX) * 2.0 - 1.0
        lam_raw_expected = 0.0
        rho_raw_expected = 0.25
        direct_norm_t = torch.tensor([[v_raw_expected, lam_raw_expected, rho_raw_expected]], dtype=torch.float32)

        with torch.no_grad():
            q1_from_norm = critic.q1_forward(state_t, act_norm_t).item()
            q1_from_direct = critic.q1_forward(state_t, direct_norm_t).item()

        results.append({
            "v_phys": v_phys,
            "v_raw": norm_arr[0],
            "q1_via_normalize": q1_from_norm,
            "q1_via_direct": q1_from_direct,
            "match": math.isclose(q1_from_norm, q1_from_direct, abs_tol=1e-6),
        })
    return results


def main():
    checkpoint_dir = Path("checkpoints/diag_actionscale_fix")
    checkpoints = [
        ("ep200", checkpoint_dir / "td3_agent_ep200.pt"),
        ("ep400", checkpoint_dir / "td3_agent_ep400.pt"),
        ("ep600", checkpoint_dir / "td3_agent_ep600.pt"),
        ("ep800", checkpoint_dir / "td3_agent_ep800.pt"),
        ("final", checkpoint_dir / "td3_agent_final.pt"),
    ]

    print("=" * 90)
    print("ACTION-SCALE FIX DIAGNOSTIC EVALUATION")
    print("=" * 90)

    # 1. Behavioral rollouts across checkpoints
    for name, path in checkpoints:
        if not path.is_file():
            print(f"Skipping {name}: {path} not found.")
            continue

        records = evaluate_checkpoint_behavior(str(path), num_seeds=10)
        mean_reward = np.mean([r["reward"] for r in records])
        mean_final_disp = np.mean([r["final_disp"] for r in records])
        mean_max_dist = np.mean([r["max_dist"] for r in records])
        exceed_count = sum(1 for r in records if r["exceeds_50m"])

        print(f"\n--- Checkpoint: {name} ({path.name}) ---")
        print(f"Displacement: Mean Final = {mean_final_disp:.1f}m | Mean Max = {mean_max_dist:.1f}m | Seeds Exceeding 50m: {exceed_count} / 10")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"{'Seed':<5} | {'Final Disp':<12} | {'Max Dist':<12} | {'Exceed 50m?':<12} | {'Reward':<12}")
        print("-" * 55)
        for r in records:
            print(f"{r['seed']:<5} | {r['final_disp']:<12.1f} | {r['max_dist']:<12.1f} | {('YES' if r['exceeds_50m'] else 'NO'):<12} | {r['reward']:<12.2f}")

    # 2. Critic sanity and consistency check on final checkpoint
    final_path = checkpoint_dir / "td3_agent_final.pt"
    if final_path.is_file():
        test_env = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0))
        s0 = test_env.reset()

        print("\n" + "=" * 90)
        print("CRITIC SANITY & CONSISTENCY CHECK (td3_agent_final.pt on s0, heading diagonally toward goal)")
        print("Comparing Q1 evaluated via normalize_action(phys) vs direct normalized tensor:")
        print("=" * 90)
        q_results = evaluate_critic_consistency(str(final_path), s0)
        print(f"{'Speed (m/s)':<12} | {'v_norm':<8} | {'Q1 via normalize_action':<24} | {'Q1 via direct tensor':<22} | {'Exact Match?':<12}")
        print("-" * 85)
        for q in q_results:
            match_str = "YES (Identical)" if q["match"] else "NO (MISMATCH)"
            print(f"{q['v_phys']:<12.1f} | {q['v_raw']:<8.2f} | {q['q1_via_normalize']:<24.4f} | {q['q1_via_direct']:<22.4f} | {match_str:<12}")


if __name__ == "__main__":
    main()
