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
    M_EPISODES,
    REPLAY_SIZE,
    R_RAND,
)
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import select_action
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
    agent = TD3Agent(state_dim=state_dim)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=3, capacity=REPLAY_SIZE)

    os.makedirs(checkpoint_dir, exist_ok=True)
    episode_rewards: List[float] = []

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
        state = env.reset()
        episode_reward = 0.0
        done = False

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

            replay_buffer.add(
                state=state,
                action=np.array(action, dtype=np.float32),
                reward=reward,
                next_state=next_state,
                done=done,
            )
            replay_experience_count += 1

            # Training trigger (Algorithm 1, Line 17):
            # Perform gradient descent only after R_ex > R_rand AND enough samples exist for a batch.
            if replay_experience_count > r_rand and len(replay_buffer) >= batch_size:
                agent.train_step(replay_buffer, batch_size, rng)

            state = next_state
            episode_reward += reward

        episode_rewards.append(float(episode_reward))

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
        use_progress_bar=not args.no_progress,
        resume=args.resume,
        resume_from=args.resume_from,
        charge_energy_on_cancelled_move=not args.no_charge_on_cancel,
    )
