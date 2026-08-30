"""
Proximal Policy Optimization (PPO) Training Script (M12 Baseline).

Trains a continuous Gaussian PPO agent on the UAV trajectory optimization task.

Key differences from PKTD3-TD (M9):
  1. Policy: Stochastic Gaussian, no prior-knowledge phase (epsilon-greedy-like
     exploration via policy entropy is inherent to PPO; no separate warmup).
  2. Training loop: Rollout-based (collect ROLLOUT_LENGTH steps, then update),
     NOT episode-based with a replay buffer.
  3. Network: Separate actor (PPOActor) and critic (PPOCritic), no target networks.
  4. Update: PPO-Clip with GAE-Lambda advantages, multi-epoch minibatch updates.

Loop structure (DESIGN DECISION: rollout_length=2048, standard PPO default):
    while total_steps < budget:
        collect ROLLOUT_LENGTH env steps (resetting env on episode end)
        compute GAE advantages
        run update_epochs minibatch epochs over collected rollout
        repeat
"""

from __future__ import annotations

import argparse
import json
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

from uav_trajectory_rl.baselines.ppo import PPOAgent, RolloutBuffer
from uav_trajectory_rl.config import (
    GAMMA,
    LEARNING_RATE,
    M_EPISODES,
    N_SLOTS,
)
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# DESIGN DECISION: PPO is driven by total steps and rollout cycles, not episodes.
# Default rollout_length=2048 is the standard PPO default (Schulman et al., 2017,
# and OpenAI Baselines). Converting from an episode budget: 800 episodes * 200
# steps/episode = 160,000 total steps (worst-case; actual episodes end early on
# arrival). This preserves comparability with the 800-episode Dueling DQL run.
DEFAULT_ROLLOUT_LENGTH = 2048


def main(
    total_steps: Optional[int] = None,
    num_episodes: Optional[int] = None,
    k_users: int = 10,
    seed: int = 0,
    rollout_length: int = DEFAULT_ROLLOUT_LENGTH,
    checkpoint_dir: str = "checkpoints/ppo",
    checkpoint_every: int = DEFAULT_ROLLOUT_LENGTH * 10,  # every ~10 rollouts
    log_every: int = 10,
    lr: float = LEARNING_RATE,
    gamma: float = GAMMA,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    update_epochs: int = 10,
    minibatch_size: int = 64,
    use_progress_bar: bool = True,
) -> List[float]:
    """
    Execute the PPO training loop.

    Parameters:
        total_steps:    Total environment steps to train (primary budget).
        num_episodes:   Optional episode budget; converted to total_steps if set
                        (assumes max N_SLOTS steps per episode for worst-case budget).
                        If both are None, defaults to 800-equivalent episodes.
        k_users:        Number of ground users K.
        seed:           Random seed for reproducibility.
        rollout_length: Steps per rollout collection cycle (DESIGN DECISION: 2048).
        checkpoint_dir: Directory for checkpoints and logs.
        checkpoint_every: Save a checkpoint every this many total steps.
        log_every:      Log episode results every N episodes.
        lr:             Adam learning rate.
        gamma:          Discount factor.
        gae_lambda:     GAE-Lambda smoothing (DESIGN DECISION: 0.95).
        clip_eps:       PPO-Clip epsilon (DESIGN DECISION: 0.2).
        value_coef:     Value loss coefficient (DESIGN DECISION: 0.5).
        entropy_coef:   Entropy bonus coefficient (DESIGN DECISION: 0.01).
        update_epochs:  Epochs per rollout update (DESIGN DECISION: 10).
        minibatch_size: Minibatch size for each epoch (DESIGN DECISION: 64).
        use_progress_bar: Whether to display tqdm progress bar.

    Returns:
        List of per-episode cumulative rewards.
    """
    # Resolve step budget
    if total_steps is None and num_episodes is None:
        num_episodes = 800
    if total_steps is None:
        # Convert episodes to steps conservatively (max steps/episode)
        total_steps = int(num_episodes * N_SLOTS)

    # 1. Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 2. Instantiate MDP environment
    env = UAVTrajectoryEnv(k=k_users, rng=rng)
    state_dim = env.state_dim

    # 3. Instantiate PPO agent and rollout buffer
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=3,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        lr=lr,
    )
    rollout_buffer = RolloutBuffer(
        rollout_length=rollout_length,
        state_dim=state_dim,
        action_dim=3,
    )

    print("=" * 70)
    print("STARTING PPO BASELINE TRAINING (M12)")
    print(f"Total steps: {total_steps} | K: {k_users} | Seed: {seed}")
    print(f"Rollout length: {rollout_length} | LR: {lr} | gamma: {gamma}")
    print(f"GAE-lambda: {gae_lambda} | clip_eps: {clip_eps} | update_epochs: {update_epochs}")
    print(f"Checkpoint Dir: {ckpt_path.resolve()}")
    print("=" * 70)

    # Tracking
    episode_rewards: List[float] = []
    episode_stats: List[dict] = []

    global_step = 0
    episode_num = 0
    last_ckpt_step = 0

    # Episode-level state (spans rollout boundaries)
    state = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    ep_arrived = False

    # Progress bar (step-based)
    pbar = None
    if use_progress_bar and tqdm is not None:
        pbar = tqdm(total=total_steps, desc="PPO Training", unit="step", dynamic_ncols=True)

    # 4. Main rollout loop
    while global_step < total_steps:
        rollout_buffer.reset()

        # --- Collect one rollout ---
        for _t in range(rollout_length):
            if global_step >= total_steps:
                break

            value = agent.get_value(state)
            physical_action, raw_action, log_prob = agent.select_action(state)

            next_state, reward, done, info = env.step(physical_action)
            global_step += 1
            ep_steps += 1
            ep_reward += reward

            if info.get("arrived", False):
                ep_arrived = True

            rollout_buffer.add(state, raw_action, log_prob, reward, value, done)

            state = next_state

            if done:
                # Episode ended mid-rollout: log it
                episode_num += 1
                episode_rewards.append(float(ep_reward))
                episode_stats.append({
                    "episode": episode_num,
                    "steps": ep_steps,
                    "arrived": ep_arrived,
                    "reward": float(ep_reward),
                    "total_steps": global_step,
                })

                if episode_num % log_every == 0:
                    window = min(len(episode_rewards), log_every)
                    recent_avg = float(np.mean(episode_rewards[-window:]))
                    msg = (
                        f"Episode {episode_num:4d} | step={global_step:7d} | "
                        f"reward={ep_reward:+8.3f} | avg({window})={recent_avg:+8.3f} | "
                        f"steps_this_ep={ep_steps}"
                    )
                    if pbar is not None:
                        pbar.write(msg)
                    else:
                        print(msg)

                # Reset episode state
                state = env.reset()
                ep_reward = 0.0
                ep_steps = 0
                ep_arrived = False

            if pbar is not None:
                pbar.update(1)

        # --- GAE computation ---
        # Bootstrap the last value: if the rollout ended mid-episode (not done),
        # use V(s_final) as the bootstrap value; if it ended on a terminal step,
        # the last stored done=True already zeroes it out in the backward pass.
        # We compute last_value from the current state after the rollout.
        if not done:
            last_value = agent.get_value(state)
        else:
            last_value = 0.0

        rollout_buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

        # --- PPO update ---
        update_info = agent.update(rollout_buffer, rng)

        # Periodic logging of training metrics
        if pbar is not None:
            pbar.set_postfix({
                "pi_loss": f"{update_info['policy_loss']:+.3f}",
                "v_loss": f"{update_info['value_loss']:.3f}",
                "H": f"{update_info['entropy']:.3f}",
                "ep": episode_num,
            })

        # --- Periodic checkpointing (step-based) ---
        if global_step - last_ckpt_step >= checkpoint_every:
            ckpt_file = ckpt_path / f"ppo_step{global_step}.pt"
            agent.save(ckpt_file)
            np.save(ckpt_path / "episode_rewards.npy", np.array(episode_rewards, dtype=np.float32))
            with open(ckpt_path / "episode_stats.json", "w", encoding="utf-8") as f:
                json.dump(episode_stats, f, indent=2)
            last_ckpt_step = global_step

    if pbar is not None:
        pbar.close()

    # 5. Save final artifacts
    final_ckpt = ckpt_path / "ppo_final.pt"
    agent.save(final_ckpt)
    np.save(ckpt_path / "episode_rewards.npy", np.array(episode_rewards, dtype=np.float32))
    with open(ckpt_path / "episode_stats.json", "w", encoding="utf-8") as f:
        json.dump(episode_stats, f, indent=2)

    # Plot reward curve
    if plt is not None and len(episode_rewards) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label="Episode Reward", alpha=0.35, color="darkorange")
        if len(episode_rewards) >= 20:
            smoothed = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
            plt.plot(range(19, len(episode_rewards)), smoothed,
                     label="20-Episode Moving Avg", color="darkred", linewidth=1.5)
        plt.title("PPO Baseline Training Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ckpt_path / "ppo_reward_curve.png", dpi=150)
        plt.close()

    print(f"\nTraining Complete. Total episodes: {episode_num}. Artifacts: {ckpt_path.resolve()}")
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Baseline for 3D UAV Trajectory Design")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Total environment steps (primary budget)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episode budget (converted to steps via N_SLOTS)")
    parser.add_argument("--k", type=int, default=10, help="Number of ground users (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--rollout-length", type=int, default=DEFAULT_ROLLOUT_LENGTH,
                        help=f"Steps per rollout cycle (default: {DEFAULT_ROLLOUT_LENGTH})")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_ROLLOUT_LENGTH * 10,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log progress every N episodes (default: 10)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help=f"Discount factor (default: {GAMMA})")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE-Lambda (DESIGN DECISION: 0.95)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO-Clip epsilon (DESIGN DECISION: 0.2)")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value loss coefficient (DESIGN DECISION: 0.5)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient (DESIGN DECISION: 0.01)")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="PPO update epochs per rollout (DESIGN DECISION: 10)")
    parser.add_argument("--minibatch-size", type=int, default=64,
                        help="Minibatch size per epoch (DESIGN DECISION: 64)")
    parser.add_argument("--no-progress-bar", action="store_true",
                        help="Disable tqdm progress bar")

    args = parser.parse_args()

    main(
        total_steps=args.total_steps,
        num_episodes=args.episodes,
        k_users=args.k,
        seed=args.seed,
        rollout_length=args.rollout_length,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        use_progress_bar=not args.no_progress_bar,
    )
