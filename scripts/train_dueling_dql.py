"""
Dueling Deep Q-Learning (Dueling DQL) Training Script (M11 Baseline).

Trains a discrete-action Dueling Q-Network on the UAV trajectory optimization task.

Differences from PKTD3-TD (M9):
  1. Action space: Discretized into 200 physical velocity combinations (5x5x8 grid).
  2. Exploration: Epsilon-greedy decay (1.0 -> 0.05) instead of heuristic prior knowledge.
     (DESIGN DECISION: Standard DQN exploration paradigm, not paper-specified).
  3. Network: Single DuelingQNetwork with target network (no twin critic, no policy delay).
  4. Target updates: Polyak soft updates (tau=0.005) executed every gradient step.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Ensure 'src' is resolvable unconditionally
_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
import torch

from uav_trajectory_rl.baselines.dueling_dql import (
    DiscreteReplayBuffer,
    DuelingDQLAgent,
    discrete_action_to_physical,
)
from uav_trajectory_rl.config import (
    BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    M_EPISODES,
    NUM_DISCRETE_ACTIONS,
    REPLAY_SIZE,
    TAU,
)
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def main(
    num_episodes: int = M_EPISODES,
    k_users: int = 10,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
    checkpoint_dir: str = "checkpoints/dueling_dql",
    checkpoint_every: int = 500,
    log_every: int = 10,
    lr: float = LEARNING_RATE,
    gamma: float = GAMMA,
    tau: float = TAU,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: Optional[int] = None,
    replay_size: int = REPLAY_SIZE,
    use_progress_bar: bool = True,
    resume: bool = False,
    resume_from: Optional[str] = None,
) -> List[float]:
    """
    Execute the Dueling DQL training loop.

    Parameters:
        num_episodes: Total episodes to train.
        k_users: Number of ground users K.
        batch_size: Mini-batch size for gradient descent.
        seed: Random seed for reproducibility.
        checkpoint_dir: Directory where checkpoints and logs are saved.
        checkpoint_every: Save checkpoint every N episodes.
        log_every: Log progress to stdout every N episodes.
        lr: Adam learning rate.
        gamma: Discount factor.
        tau: Polyak soft target update rate.
        epsilon_start: Initial epsilon for epsilon-greedy exploration.
        epsilon_end: Minimum epsilon floor.
        epsilon_decay_episodes: Episodes over which epsilon decays linearly (default: 80% of num_episodes).
        replay_size: Maximum transitions in DiscreteReplayBuffer.
        use_progress_bar: Whether to display tqdm progress bar.
        resume: Whether to resume from latest checkpoint.
        resume_from: Explicit path to checkpoint file.
    """
    # 1. Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    if epsilon_decay_episodes is None:
        epsilon_decay_episodes = max(1, int(0.8 * num_episodes))

    # 2. Instantiate MDP environment
    env = UAVTrajectoryEnv(k=k_users, rng=rng)
    state_dim = env.state_dim

    # 3. Instantiate Agent & Replay Buffer
    agent = DuelingDQLAgent(
        state_dim=state_dim,
        num_actions=NUM_DISCRETE_ACTIONS,
        gamma=gamma,
        tau=tau,
        lr=lr,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        seed=seed,
    )
    replay_buffer = DiscreteReplayBuffer(state_dim=state_dim, capacity=replay_size)

    start_episode = 1
    episode_rewards: List[float] = []
    episode_stats: List[dict] = []

    # 4. Handle resume
    target_resume_path = None
    if resume_from is not None:
        target_resume_path = Path(resume_from)
    elif resume:
        ckpt_files = glob.glob(str(ckpt_path / "dueling_dql_ep*.pt"))
        if ckpt_files:
            def extract_ep(fname: str) -> int:
                m = re.search(r"dueling_dql_ep(\d+)\.pt", fname)
                return int(m.group(1)) if m else -1
            target_resume_path = Path(max(ckpt_files, key=extract_ep))

    if target_resume_path is not None and target_resume_path.exists():
        print(f"Resuming Dueling DQL from checkpoint: {target_resume_path}")
        agent.load(target_resume_path)
        m = re.search(r"dueling_dql_ep(\d+)\.pt", target_resume_path.name)
        if m:
            start_episode = int(m.group(1)) + 1
        rewards_npy = ckpt_path / "episode_rewards.npy"
        if rewards_npy.exists():
            episode_rewards = list(np.load(rewards_npy))
        stats_json = ckpt_path / "episode_stats.json"
        if stats_json.exists():
            with open(stats_json, "r", encoding="utf-8") as f:
                episode_stats = json.load(f)

    print("=" * 70)
    print("STARTING DUELING DQL BASELINE TRAINING (M11)")
    print(f"Episodes: {num_episodes} | K: {k_users} | Batch: {batch_size} | Seed: {seed}")
    print(f"Actions: {NUM_DISCRETE_ACTIONS} | Epsilon: {epsilon_start} -> {epsilon_end} (over {epsilon_decay_episodes} eps)")
    print(f"Checkpoint Dir: {ckpt_path.resolve()}")
    print("=" * 70)

    # 5. Training Loop
    ep_range = range(start_episode, num_episodes + 1)
    pbar = None
    if use_progress_bar and tqdm is not None:
        pbar = tqdm(
            ep_range,
            desc="Dueling DQL Training",
            unit="ep",
            dynamic_ncols=True,
            initial=start_episode - 1,
            total=num_episodes,
        )
        ep_iterator = pbar
    else:
        ep_iterator = ep_range

    total_steps = 0
    total_updates = 0

    for episode in ep_iterator:
        current_eps = agent.update_epsilon(episode)
        state = env.reset()
        episode_reward = 0.0
        step_count = 0
        arrived = False
        done = False

        while not done:
            action_idx = agent.select_action(state)
            action_phys = discrete_action_to_physical(action_idx)

            next_state, reward, done, info = env.step(action_phys)
            step_count += 1
            total_steps += 1

            if info.get("arrived", False):
                arrived = True

            replay_buffer.add(state, action_idx, reward, next_state, done)

            if len(replay_buffer) >= batch_size:
                agent.train_step(replay_buffer, batch_size, rng)
                total_updates += 1

            state = next_state
            episode_reward += reward

        episode_rewards.append(float(episode_reward))
        episode_stats.append({
            "episode": episode,
            "steps": step_count,
            "arrived": arrived,
            "reward": float(episode_reward),
            "epsilon": float(current_eps),
        })

        window_size = min(len(episode_rewards), log_every)
        recent_avg = float(np.mean(episode_rewards[-window_size:]))

        if pbar is not None:
            pbar.set_postfix({
                "reward": f"{episode_reward:+7.1f}",
                f"avg{window_size}": f"{recent_avg:+7.1f}",
                "eps": f"{current_eps:.3f}",
                "updates": total_updates,
            })

        if episode % log_every == 0:
            msg = (
                f"Episode {episode:4d}/{num_episodes:4d} | "
                f"reward={episode_reward:+8.3f} | "
                f"avg({log_every})={recent_avg:+8.3f} | "
                f"eps={current_eps:.3f} | "
                f"buf={len(replay_buffer)} | "
                f"updates={total_updates}"
            )
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)

        # Periodic checkpoint
        if episode % checkpoint_every == 0:
            ckpt_file = ckpt_path / f"dueling_dql_ep{episode}.pt"
            agent.save(ckpt_file)
            np.save(ckpt_path / "episode_rewards.npy", np.array(episode_rewards, dtype=np.float32))
            with open(ckpt_path / "episode_stats.json", "w", encoding="utf-8") as f:
                json.dump(episode_stats, f, indent=2)

    # 6. Save final model and artifacts
    final_ckpt = ckpt_path / "dueling_dql_final.pt"
    agent.save(final_ckpt)
    np.save(ckpt_path / "episode_rewards.npy", np.array(episode_rewards, dtype=np.float32))
    with open(ckpt_path / "episode_stats.json", "w", encoding="utf-8") as f:
        json.dump(episode_stats, f, indent=2)

    # Plot reward curve
    if plt is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label="Episode Reward", alpha=0.35, color="teal")
        if len(episode_rewards) >= 20:
            smoothed = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
            plt.plot(range(19, len(episode_rewards)), smoothed, label="20-Episode Moving Avg", color="darkblue", linewidth=1.5)
        plt.title("Dueling DQL Baseline Training Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ckpt_path / "dueling_dql_reward_curve.png", dpi=150)
        plt.close()

    print(f"\nTraining Complete. Artifacts saved to: {ckpt_path.resolve()}")
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dueling DQL Baseline for 3D UAV Trajectory Design")
    parser.add_argument("--episodes", type=int, default=M_EPISODES, help=f"Total training episodes (default: {M_EPISODES})")
    parser.add_argument("--k", type=int, default=10, help="Number of ground users (default: 10)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dueling_dql", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save checkpoint interval (default: 500)")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress interval (default: 10)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--gamma", type=float, default=GAMMA, help=f"Discount factor (default: {GAMMA})")
    parser.add_argument("--tau", type=float, default=TAU, help=f"Polyak soft-update rate (default: {TAU})")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon (default: 1.0)")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Epsilon floor (default: 0.05)")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=None, help="Episodes over which epsilon decays")
    parser.add_argument("--replay-size", type=int, default=REPLAY_SIZE, help=f"Replay buffer size (default: {REPLAY_SIZE})")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint file to resume from")

    args = parser.parse_args()

    main(
        num_episodes=args.episodes,
        k_users=args.k,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        replay_size=args.replay_size,
        use_progress_bar=not args.no_progress_bar,
        resume=args.resume,
        resume_from=args.resume_from,
    )
