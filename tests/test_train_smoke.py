import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path so 'scripts.train' can be resolved
# regardless of current working directory or test runner invocation path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train
from scripts.train import main
import uav_trajectory_rl.config
import uav_trajectory_rl.prior_knowledge_policy


def test_training_smoke_run(tmp_path, monkeypatch):
    """
    Run an end-to-end 5-episode training smoke test to verify all components wire together.

    Uses a low r_rand threshold (5 steps) and batch_size (8) so that the network-driven branch
    and TD3 gradient updates are actively exercised during the test run.
    """
    tiny_r_rand = 5

    # Monkeypatch R_RAND across config, policy, and training script
    monkeypatch.setattr(uav_trajectory_rl.config, "R_RAND", tiny_r_rand)
    monkeypatch.setattr(uav_trajectory_rl.prior_knowledge_policy, "R_RAND", tiny_r_rand)
    monkeypatch.setattr(scripts.train, "R_RAND", tiny_r_rand)

    checkpoint_dir = str(tmp_path / "checkpoints")

    # Run 5 episodes with small user count K=3 and batch_size=8
    rewards = main(
        num_episodes=5,
        k_users=3,
        batch_size=8,
        seed=42,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=5,
        log_every=100,
        r_rand=tiny_r_rand,
    )

    # 1. Verify returns from main()
    assert len(rewards) == 5
    for r in rewards:
        assert isinstance(r, float)
        assert np.isfinite(r)

    # 2. Verify intermediate and final checkpoint files exist
    ep5_ckpt = os.path.join(checkpoint_dir, "td3_agent_ep5.pt")
    final_ckpt = os.path.join(checkpoint_dir, "td3_agent_final.pt")
    rewards_file = os.path.join(checkpoint_dir, "episode_rewards.npy")

    assert os.path.isfile(ep5_ckpt), f"Missing checkpoint: {ep5_ckpt}"
    assert os.path.isfile(final_ckpt), f"Missing final checkpoint: {final_ckpt}"
    assert os.path.isfile(rewards_file), f"Missing rewards file: {rewards_file}"

    # 3. Verify saved reward history
    saved_rewards = np.load(rewards_file)
    assert saved_rewards.shape == (5,)
    assert np.all(np.isfinite(saved_rewards))
    assert np.allclose(saved_rewards, rewards)

    # 4. Verify that the network-driven branch and train_step were actually executed
    # Safely load final checkpoint across PyTorch versions
    try:
        checkpoint_data = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint_data = torch.load(final_ckpt, map_location="cpu", weights_only=False)

    total_updates = checkpoint_data["total_updates"]
    assert total_updates > 0, f"Expected total_updates > 0, got {total_updates}"
