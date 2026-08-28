"""
Diagnostic script to evaluate UAV displacement, movement behavior,
and critic Q1-vs-speed gradients across training checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from uav_trajectory_rl.config import Q_START, V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import unnormalize_action
from uav_trajectory_rl.td3_networks import Actor, TwinCritic


def evaluate_policy_displacement(actor: Actor, num_seeds: int = 10, charge_on_cancel: bool = True):
    """
    Run full episodes across multiple seeds and track displacement from Q_START.
    """
    records = []
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)
        env = UAVTrajectoryEnv(k=10, rng=rng, charge_energy_on_cancelled_move=charge_on_cancel)
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
        exceeds_50m = max_dist_from_start >= 50.0
        records.append({
            "seed": seed,
            "final_displacement": final_disp,
            "max_distance": max_dist_from_start,
            "exceeds_50m": exceeds_50m,
            "reward": ep_reward,
            "final_pos": env.uav_pos.round(1).tolist(),
        })
    return records


def evaluate_critic_q1_vs_speed(critic: TwinCritic, test_state: np.ndarray, lam_norm: float = 0.0, rho_norm: float = 0.0):
    """
    Evaluate critic Q1 output as commanded normalized speed varies from -1.0 (0 m/s) to +1.0 (20 m/s).
    """
    state_t = torch.as_tensor(test_state, dtype=torch.float32).unsqueeze(0)
    v_raw_values = np.linspace(-1.0, 1.0, 9)
    q1_values = []

    for v_raw in v_raw_values:
        action_t = torch.tensor([[v_raw, lam_norm, rho_norm]], dtype=torch.float32)
        with torch.no_grad():
            q1 = critic.q1_forward(state_t, action_t).item()
        v_phys = (v_raw + 1.0) / 2.0 * V_MAX
        q1_values.append((v_raw, v_phys, q1))

    return q1_values


def load_actor_critic(ckpt_path: str, state_dim: int = 26, action_dim: int = 3):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=1.0)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim)
    critic.load_state_dict(ckpt["critic"])
    critic.eval()
    return actor, critic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate displacement and critic Q1 vs speed")
    parser.add_argument("--ckpt-on", type=str, default="checkpoints/diag_chargeon/td3_agent_final.pt")
    parser.add_argument("--ckpt-off", type=str, default="checkpoints/diag_chargeoff/td3_agent_final.pt")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    print("=" * 80)
    print("DIAGNOSTIC BEHAVIORAL DISPLACEMENT COMPARISON")
    print(f"Run A (Charge ON):  {args.ckpt_on}")
    print(f"Run B (Charge OFF): {args.ckpt_off}")
    print("=" * 80)

    # Initial test state from fresh env
    test_env = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0))
    s0 = test_env.reset()

    # Evaluate Run A
    if os.path.isfile(args.ckpt_on):
        actor_a, critic_a = load_actor_critic(args.ckpt_on)
        rec_a = evaluate_policy_displacement(actor_a, num_seeds=args.seeds, charge_on_cancel=True)
        q1_a = evaluate_critic_q1_vs_speed(critic_a, s0)
    else:
        rec_a, q1_a = None, None
        print(f"File not found: {args.ckpt_on}")

    # Evaluate Run B
    if os.path.isfile(args.ckpt_off):
        actor_b, critic_b = load_actor_critic(args.ckpt_off)
        rec_b = evaluate_policy_displacement(actor_b, num_seeds=args.seeds, charge_on_cancel=False)
        q1_b = evaluate_critic_q1_vs_speed(critic_b, s0)
    else:
        rec_b, q1_b = None, None
        print(f"File not found: {args.ckpt_off}")

    if rec_a and rec_b:
        print("\n--- SIDE-BY-SIDE ROLLOUT DISPLACEMENT (10 SEEDS) ---")
        print(f"{'Seed':<5} | {'Run A (Charge ON) Final/Max Disp':<32} | {'Exceed 50m?':<12} | {'Run B (Charge OFF) Final/Max Disp':<33} | {'Exceed 50m?':<12}")
        print("-" * 105)
        for sa, sb in zip(rec_a, rec_b):
            disp_a_str = f"{sa['final_displacement']:.1f}m / {sa['max_distance']:.1f}m"
            disp_b_str = f"{sb['final_displacement']:.1f}m / {sb['max_distance']:.1f}m"
            exc_a = "YES" if sa["exceeds_50m"] else "NO"
            exc_b = "YES" if sb["exceeds_50m"] else "NO"
            print(f"{sa['seed']:<5} | {disp_a_str:<32} | {exc_a:<12} | {disp_b_str:<33} | {exc_b:<12}")

        print("\n--- REWARD COMPARISON ---")
        print(f"{'Seed':<5} | {'Run A Reward':<16} | {'Run B Reward':<16}")
        print("-" * 45)
        for sa, sb in zip(rec_a, rec_b):
            print(f"{sa['seed']:<5} | {sa['reward']:<16.2f} | {sb['reward']:<16.2f}")

    if q1_a and q1_b:
        print("\n--- CRITIC Q1(s0, action) vs SPEED (Holding lam=0, rho=0) ---")
        print(f"{'v_raw':<8} | {'Speed (m/s)':<12} | {'Run A Q1 (Charge ON)':<22} | {'Run B Q1 (Charge OFF)':<22}")
        print("-" * 72)
        for (v_raw, v_phys, q1a), (_, _, q1b) in zip(q1_a, q1_b):
            print(f"{v_raw:<8.2f} | {v_phys:<12.1f} | {q1a:<22.4f} | {q1b:<22.4f}")
