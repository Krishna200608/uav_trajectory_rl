"""
PKTD3-TD Training Script (Algorithm 1).

This script implements the full 32-line training procedure of the PKTD3-TD algorithm
as published in the IEEE TNSE paper:
    "3-D Trajectory Design Based on Deep Reinforcement Learning for
     UAV-Assisted Communication Networks"

Execution Flow:
    1. Initialize MDP environment (UAVTrajectoryEnv), TD3Agent, and ReplayBuffer.
    2. For each episode m = 1..M (Table III: 6000 episodes):
       - Reset UAV position to q_s and sample initial ground user swarm locations.
       - For each time slot n = 1..N:
         - Action selection via prior-knowledge policy (M6, eq. 31):
           If total transitions R_ex <= R_rand (20,000): sample heuristic prior-knowledge action.
           If R_ex > R_rand: query TD3 actor with target smoothing noise.
         - Execute action in UAVTrajectoryEnv: apply acceleration constraint, spatial bounds,
           update user mobility, and calculate 6-part reward r_n (eq. 21-29).
         - Store transition (s_n, a_n, r_n, s_{n+1}, done) into ReplayBuffer.
         - If R_ex > R_rand and buffer has >= batch_size transitions:
           - Update twin critics via clipped double-Q Bellman targets (eq. 32, 34).
           - Every d time steps (policy_delay=2): update actor via policy gradient (eq. 33)
             and soft-update target networks (eq. 35).
    3. Save periodic checkpoints and final model weights along with training reward history.
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Ensure 'src' is resolvable unconditionally regardless of invocation mode
_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch

from uav_trajectory_rl.config import (
    BATCH_SIZE,
    GAMMA,
    M_EPISODES,
    REPLAY_SIZE,
    R_RAND,
)
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import normalize_action, select_action
from uav_trajectory_rl.td3_agent import TD3Agent
from uav_trajectory_rl.td3_networks import ReplayBuffer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def main(
    num_episodes: int = M_EPISODES,
    k_users: int = 10,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 500,
    log_every: int = 10,
    r_rand: int = R_RAND,
    use_progress_bar: bool = True,
    resume: bool = False,
    resume_from: Optional[str] = None,
    charge_energy_on_cancelled_move: bool = True,
    gamma: float = GAMMA,
    arrived_fraction: Optional[float] = None,
    terminal_window: Optional[int] = None,
) -> List[float]:
    """
    Execute the PKTD3-TD training procedure (Algorithm 1).

    Parameters:
        num_episodes: Total training episodes M (default: M_EPISODES = 6000).
        k_users: Number of ground users K (default: 10).
        batch_size: Mini-batch size for TD3 updates (default: BATCH_SIZE = 128).
        seed: Random seed for NumPy and PyTorch reproducibility.
        checkpoint_dir: Directory path where model checkpoints and logs are saved.
        checkpoint_every: Interval of episodes between saving intermediate checkpoints.
        log_every: Interval of episodes between logging progress to stdout.
        r_rand: Exploration threshold R_rand for prior-knowledge guidance (default: R_RAND = 20000).
        use_progress_bar: Whether to display interactive tqdm progress bar with ETA.
        resume: Whether to automatically resume from the latest checkpoint in checkpoint_dir.
        resume_from: Explicit path to a checkpoint file (.pt) to resume training from.
        charge_energy_on_cancelled_move: Whether to charge energy on cancelled boundary moves.
        gamma: Discount factor gamma for Bellman updates (default: GAMMA = 0.96).
        arrived_fraction: Optional fraction of batch to draw from arrived episodes (stratified sampling).
        terminal_window: Optional window size in steps from episode end for arrived transitions.

    Returns:
        List[float]: Cumulative scalar reward achieved in each episode.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = UAVTrajectoryEnv(
        k=k_users,
        rng=rng,
        charge_energy_on_cancelled_move=charge_energy_on_cancelled_move,
    )
    state_dim = env.state_dim
    agent = TD3Agent(state_dim=state_dim, gamma=gamma)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=3, capacity=REPLAY_SIZE)

    os.makedirs(checkpoint_dir, exist_ok=True)
    episode_rewards: List[float] = []
    episode_stats: List[dict] = []

    # ==========================================================================
    # IMPORTANT DESIGN NOTE ON R_ex (replay_experience_count):
    # R_ex represents the total cumulative transition steps EVER added to the
    # replay buffer (monotonically increasing counter), NOT len(replay_buffer)
    # (which saturates at REPLAY_SIZE once circular overwrite begins).
    # This matches the paper's formal semantics in eq. (31) and Algorithm 1 Line 17.
    # ==========================================================================
    replay_experience_count: int = 0
    start_episode: int = 1

    # Checkpoint resumption logic
    if resume or resume_from is not None:
        target_ckpt = None
        if resume_from is not None and os.path.isfile(resume_from):
            target_ckpt = resume_from
        else:
            state_file = os.path.join(checkpoint_dir, "training_state.json")
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        meta = json.load(f)
                    last_ep = meta.get("last_episode", 0)
                    candidate = os.path.join(checkpoint_dir, f"td3_agent_ep{last_ep}.pt")
                    if os.path.isfile(candidate):
                        target_ckpt = candidate
                except Exception:
                    pass
            if target_ckpt is None:
                ckpts = glob.glob(os.path.join(checkpoint_dir, "td3_agent_ep*.pt"))
                if ckpts:
                    target_ckpt = max(
                        ckpts,
                        key=lambda f: int(re.search(r"ep(\d+)", f).group(1)) if re.search(r"ep(\d+)", f) else 0,
                    )

        if target_ckpt and os.path.isfile(target_ckpt):
            agent.load(target_ckpt)
            m = re.search(r"ep(\d+)", os.path.basename(target_ckpt))
            last_ep = int(m.group(1)) if m else 0
            start_episode = last_ep + 1

            state_file = os.path.join(checkpoint_dir, "training_state.json")
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        meta = json.load(f)
                    replay_experience_count = int(meta.get("replay_experience_count", last_ep * 200))
                except Exception:
                    replay_experience_count = last_ep * 200
            else:
                replay_experience_count = last_ep * 200

            rewards_file = os.path.join(checkpoint_dir, "episode_rewards.npy")
            if os.path.exists(rewards_file):
                try:
                    saved_arr = np.load(rewards_file)
                    episode_rewards = list(saved_arr[:last_ep])
                except Exception:
                    episode_rewards = []

    print("=" * 70)
    print("PKTD3-TD Training Initialization")
    print(f"  Episodes: {start_episode}..{num_episodes} | Users K: {k_users} | State Dim: {state_dim}")
    print(f"  Batch Size: {batch_size} | Buffer Capacity: {REPLAY_SIZE} | R_rand: {r_rand}")
    print(f"  Checkpoints: {checkpoint_dir} (every {checkpoint_every} eps)")
    if start_episode > 1:
        print(f"  [RESUME] Active: Continuing from Episode {start_episode} (R_ex={replay_experience_count})")
    print("=" * 70)

    ep_iterator = range(start_episode, num_episodes + 1)
    pbar = None
    if use_progress_bar and tqdm is not None:
        pbar = tqdm(
            ep_iterator,
            desc="PKTD3-TD Training",
            unit="ep",
            dynamic_ncols=True,
            initial=start_episode - 1,
            total=num_episodes,
        )
        ep_iterator = pbar

    for episode in ep_iterator:
        is_network_phase = (replay_experience_count > r_rand)
        state = env.reset()
        episode_reward = 0.0
        step_count = 0
        arrived = False
        done = False
        episode_transitions = []

        while not done:
            # Action selection (eq. 31 via M6's dispatcher):
            # Passes agent.select_action as actor_fn (normalized [-c, c]^3 output contract).
            action = select_action(
                state=state,
                replay_buffer_size=replay_experience_count,
                actor_fn=agent.select_action,
                rng=rng,
                r_rand=r_rand,
            )
            next_state, reward, done, info = env.step(action)
            step_count += 1
            if info.get("arrived", False):
                arrived = True

            normalized_action = normalize_action(action)
            episode_transitions.append((
                state,
                np.array(normalized_action, dtype=np.float32),
                reward,
                next_state,
                done,
            ))
            state = next_state
            episode_reward += reward

        # Backfill: Add all transitions from this episode tagged with its final arrived outcome
        num_trans = len(episode_transitions)
        for i, (s, a, r, ns, d) in enumerate(episode_transitions):
            steps_from_terminal = num_trans - 1 - i
            replay_buffer.add(
                state=s,
                action=a,
                reward=r,
                next_state=ns,
                done=d,
                arrived=arrived,
                steps_from_terminal=steps_from_terminal,
            )
            replay_experience_count += 1

            # Training trigger (Algorithm 1, Line 17):
            # Perform gradient descent only after R_ex > R_rand AND enough samples exist for a batch.
            if replay_experience_count > r_rand and len(replay_buffer) >= batch_size:
                agent.train_step(
                    replay_buffer,
                    batch_size,
                    rng,
                    arrived_fraction=arrived_fraction,
                    terminal_window=terminal_window,
                )

        episode_rewards.append(float(episode_reward))
        episode_stats.append({
            "episode": episode,
            "steps": step_count,
            "arrived": arrived,
            "network_phase": is_network_phase,
            "reward": float(episode_reward),
        })

        # Recent rolling average
        window_size = min(len(episode_rewards), log_every)
        recent_avg = float(np.mean(episode_rewards[-window_size:]))

        if pbar is not None:
            pbar.set_postfix(
                {
                    "reward": f"{episode_reward:+7.1f}",
                    f"avg{window_size}": f"{recent_avg:+7.1f}",
                    "R_ex": replay_experience_count,
                    "updates": agent.total_updates,
                }
            )

        if episode % log_every == 0:
            log_msg = (
                f"Episode {episode:4d}/{num_episodes:4d} | "
                f"reward={episode_reward:+8.3f} | "
                f"avg(last {log_every:2d})={recent_avg:+8.3f} | "
                f"R_ex={replay_experience_count} | "
                f"updates={agent.total_updates}"
            )
            if pbar is not None:
                pbar.write(log_msg)
            else:
                print(log_msg, flush=True)

        if episode % checkpoint_every == 0:
            ckpt_file = os.path.join(checkpoint_dir, f"td3_agent_ep{episode}.pt")
            agent.save(ckpt_file)
            # Intermediate save of reward array so external monitors / Drive can track live progress
            rewards_file = os.path.join(checkpoint_dir, "episode_rewards.npy")
            np.save(rewards_file, np.array(episode_rewards, dtype=np.float32))

            # Save state metadata for seamless resumption
            state_file = os.path.join(checkpoint_dir, "training_state.json")
            try:
                with open(state_file, "w") as f:
                    json.dump(
                        {
                            "last_episode": episode,
                            "replay_experience_count": replay_experience_count,
                            "total_updates": agent.total_updates,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass

            ckpt_msg = f"  --> Checkpoint saved: {ckpt_file}"
            if pbar is not None:
                pbar.write(ckpt_msg)
            else:
                print(ckpt_msg, flush=True)

    if pbar is not None:
        pbar.close()

    # Save final model weights and reward trajectory
    final_ckpt = os.path.join(checkpoint_dir, "td3_agent_final.pt")
    agent.save(final_ckpt)
    rewards_file = os.path.join(checkpoint_dir, "episode_rewards.npy")
    np.save(rewards_file, np.array(episode_rewards, dtype=np.float32))

    # Save detailed episode statistics (steps, arrival, network_phase)
    stats_file = os.path.join(checkpoint_dir, "episode_stats.json")
    try:
        with open(stats_file, "w") as f:
            json.dump(episode_stats, f, indent=2)
    except Exception:
        pass

    net_eps = [e for e in episode_stats if e["network_phase"]]
    if net_eps:
        arrived_net = [e for e in net_eps if e["arrived"]]
        non_arrived_net = [e for e in net_eps if not e["arrived"]]
        arr_rate = len(arrived_net) / len(net_eps) * 100.0
        mean_steps_arr = float(np.mean([e["steps"] for e in arrived_net])) if arrived_net else 0.0
        mean_steps_non = float(np.mean([e["steps"] for e in non_arrived_net])) if non_arrived_net else 0.0
        trans_arr = sum(e["steps"] for e in arrived_net)
        trans_non = sum(e["steps"] for e in non_arrived_net)
        trans_total = trans_arr + trans_non
        frac_arr_trans = trans_arr / trans_total * 100.0 if trans_total > 0 else 0.0

        print("\n" + "=" * 70, flush=True)
        print("REPLAY BUFFER COMPOSITION (NETWORK-PHASE EPISODES ONLY)", flush=True)
        print("=" * 70, flush=True)
        print(f"Total Network-Phase Episodes: {len(net_eps)}", flush=True)
        print(f"Arrived Network-Phase Episodes: {len(arrived_net)} ({arr_rate:.1f}%)", flush=True)
        print(f"Mean Steps (Arrived Episodes): {mean_steps_arr:.1f}", flush=True)
        print(f"Mean Steps (Non-Arrived Episodes): {mean_steps_non:.1f}", flush=True)
        print(f"Total Transitions in Buffer: {trans_total}", flush=True)
        print(f"  - From Arrived Episodes:     {trans_arr} ({frac_arr_trans:.2f}%)", flush=True)
        print(f"  - From Non-Arrived Episodes: {trans_non} ({100.0 - frac_arr_trans:.2f}%)", flush=True)
        print("=" * 70, flush=True)

    print(f"Training complete. Final checkpoint saved to: {final_ckpt}", flush=True)
    return episode_rewards


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PKTD3-TD training."""
    parser = argparse.ArgumentParser(description="PKTD3-TD Training Loop (Algorithm 1)")
    parser.add_argument("--episodes", type=int, default=M_EPISODES, help="Total episodes to train")
    parser.add_argument("--k", type=int, default=10, help="Number of ground users")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint saving interval in episodes")
    parser.add_argument("--log-every", type=int, default=10, help="Logging interval in episodes")
    parser.add_argument("--r-rand", type=int, default=R_RAND, help="Prior knowledge exploration threshold R_rand")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor gamma for Bellman updates (default: 0.96)")
    parser.add_argument(
        "--arrived-fraction",
        type=float,
        default=None,
        help="Fraction of mini-batch sampled from arrived episodes (stratified replay sampling, default: None = uniform)",
    )
    parser.add_argument(
        "--terminal-window",
        type=int,
        default=None,
        help="Maximum steps from episode end for arrived transitions (terminal-weighted replay sampling)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable visual tqdm progress bar")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint in checkpoint-dir")
    parser.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint file (.pt) to resume from")
    parser.add_argument(
        "--no-charge-on-cancel",
        action="store_true",
        help="Zero out speed in energy calculation when a move is cancelled by spatial boundaries (diagnostic hypothesis test)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        num_episodes=args.episodes,
        k_users=args.k,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        r_rand=args.r_rand,
        gamma=args.gamma,
        arrived_fraction=args.arrived_fraction,
        terminal_window=args.terminal_window,
        use_progress_bar=not args.no_progress,
        resume=args.resume,
        resume_from=args.resume_from,
        charge_energy_on_cancelled_move=not args.no_charge_on_cancel,
    )
