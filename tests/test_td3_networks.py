import numpy as np
import pytest
import torch

from uav_trajectory_rl.config import ACTION_CLIP_C
from uav_trajectory_rl.td3_networks import (
    Actor,
    ReplayBuffer,
    TwinCritic,
    to_torch_batch,
)


def test_actor_forward_shape_and_bounds():
    state_dim = 26  # 2 * K + 6 for K=10
    action_dim = 3
    batch_size = 4
    c = ACTION_CLIP_C  # 1.0

    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=c)
    actor.eval()

    # Forward pass with a mini-batch (created on actor.device)
    dummy_states = torch.randn(batch_size, state_dim, device=actor.device)
    with torch.no_grad():
        actions = actor(dummy_states)

    assert actions.shape == (batch_size, action_dim)
    assert (actions >= -c).all() and (actions <= c).all()

    # Empirical stress test across 100 random state vectors
    stress_states = torch.randn(100, state_dim, device=actor.device) * 500.0  # large dynamic range
    with torch.no_grad():
        stress_actions = actor(stress_states)

    assert stress_actions.shape == (100, action_dim)
    assert (stress_actions >= -c).all() and (stress_actions <= c).all()

    # Cross-device test: passing CPU tensor to actor must work seamlessly via internal .to(self.device)
    cpu_states = torch.randn(batch_size, state_dim, device="cpu")
    with torch.no_grad():
        cpu_actions = actor(cpu_states)
    assert cpu_actions.shape == (batch_size, action_dim)


def test_twin_critic_forward_and_q1_consistency():
    state_dim = 26
    action_dim = 3
    batch_size = 4

    critic = TwinCritic(state_dim=state_dim, action_dim=action_dim)
    critic.eval()

    dummy_states = torch.randn(batch_size, state_dim, device=critic.device)
    dummy_actions = torch.randn(batch_size, action_dim, device=critic.device)

    with torch.no_grad():
        q1, q2 = critic(dummy_states, dummy_actions)
        q1_direct = critic.q1_forward(dummy_states, dummy_actions)

    assert q1.shape == (batch_size, 1)
    assert q2.shape == (batch_size, 1)
    assert q1_direct.shape == (batch_size, 1)

    # q1 from forward() and q1_forward() must match exactly for identical inputs
    assert torch.allclose(q1, q1_direct, atol=1e-7)

    # Cross-device test: passing CPU tensor to critic must work seamlessly
    cpu_states = torch.randn(batch_size, state_dim, device="cpu")
    cpu_actions = torch.randn(batch_size, action_dim, device="cpu")
    with torch.no_grad():
        q1_cpu, q2_cpu = critic(cpu_states, cpu_actions)
    assert q1_cpu.shape == (batch_size, 1)


def test_replay_buffer_capacity_and_circular_overwrite():
    capacity = 5
    state_dim = 4
    action_dim = 2
    buf = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=capacity)

    # 1. Add fewer than capacity
    for i in range(3):
        buf.add(
            state=np.ones(state_dim) * i,
            action=np.ones(action_dim) * i,
            reward=float(i),
            next_state=np.ones(state_dim) * (i + 1),
            done=False,
        )
    assert len(buf) == 3

    # 2. Add up to capacity
    for i in range(3, 5):
        buf.add(
            state=np.ones(state_dim) * i,
            action=np.ones(action_dim) * i,
            reward=float(i),
            next_state=np.ones(state_dim) * (i + 1),
            done=False,
        )
    assert len(buf) == 5

    # 3. Add more than capacity (add 3 more, total 8 additions)
    # The rewards added in order: 0, 1, 2, 3, 4, 5, 6, 7
    # Since capacity=5, slots [0, 1, 2] will be overwritten by 5, 6, 7.
    # Stored rewards should be {3, 4, 5, 6, 7}.
    for i in range(5, 8):
        buf.add(
            state=np.ones(state_dim) * i,
            action=np.ones(action_dim) * i,
            reward=float(i),
            next_state=np.ones(state_dim) * (i + 1),
            done=False,
        )
    assert len(buf) == 5

    stored_rewards = set(buf.rewards.flatten().tolist())
    assert stored_rewards == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_replay_buffer_sample():
    capacity = 10
    state_dim = 3
    action_dim = 2
    buf = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=capacity)
    rng = np.random.default_rng(42)

    # Fill buffer with 6 transitions
    for i in range(6):
        buf.add(
            state=np.full(state_dim, i, dtype=np.float32),
            action=np.full(action_dim, i, dtype=np.float32),
            reward=float(i),
            next_state=np.full(state_dim, i + 1, dtype=np.float32),
            done=(i == 5),
        )

    # Sample valid batch size
    batch_size = 4
    states, actions, rewards, next_states, dones = buf.sample(batch_size, rng)

    assert states.shape == (batch_size, state_dim)
    assert actions.shape == (batch_size, action_dim)
    assert rewards.shape == (batch_size, 1)
    assert next_states.shape == (batch_size, state_dim)
    assert dones.shape == (batch_size, 1)

    # Sample exceeding current size should raise ValueError
    with pytest.raises(ValueError, match="Cannot sample 10 transitions"):
        buf.sample(10, rng)


def test_to_torch_batch():
    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr2 = np.array([[0.5], [1.5]], dtype=np.float32)

    t1, t2 = to_torch_batch(arr1, arr2, device=torch.device("cpu"))

    assert isinstance(t1, torch.Tensor)
    assert isinstance(t2, torch.Tensor)
    assert t1.dtype == torch.float32
    assert t2.dtype == torch.float32
    assert t1.device.type == "cpu"
    assert np.allclose(t1.numpy(), arr1)
    assert np.allclose(t2.numpy(), arr2)


def test_replay_buffer_sample_stratified():
    """
    Test that sample_stratified draws the exact target fraction of arrived transitions.
    """
    capacity = 20
    state_dim = 3
    action_dim = 2
    buf = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=capacity)
    rng = np.random.default_rng(123)

    # Add 4 arrived transitions (marked by reward >= 1000.0)
    for i in range(4):
        buf.add(
            state=np.full(state_dim, float(i), dtype=np.float32),
            action=np.full(action_dim, 1.0, dtype=np.float32),
            reward=1000.0 + float(i),
            next_state=np.full(state_dim, float(i + 1), dtype=np.float32),
            done=True,
            arrived=True,
        )

    # Add 16 non-arrived transitions (marked by reward < 100.0)
    for i in range(16):
        buf.add(
            state=np.full(state_dim, float(i), dtype=np.float32),
            action=np.full(action_dim, -1.0, dtype=np.float32),
            reward=float(i),
            next_state=np.full(state_dim, float(i + 1), dtype=np.float32),
            done=False,
            arrived=False,
        )

    assert len(buf) == 20

    # Sample batch of 8 with arrived_fraction=0.5 -> expect exactly 4 arrived transitions
    batch_size = 8
    states, actions, rewards, next_states, dones = buf.sample_stratified(
        batch_size=batch_size,
        arrived_fraction=0.5,
        rng=rng,
    )

    assert states.shape == (batch_size, state_dim)
    assert actions.shape == (batch_size, action_dim)
    assert rewards.shape == (batch_size, 1)

    arrived_count = int(np.sum(rewards >= 1000.0))
    non_arrived_count = int(np.sum(rewards < 100.0))
    assert arrived_count == 4, f"Expected 4 arrived transitions, got {arrived_count}"
    assert non_arrived_count == 4, f"Expected 4 non-arrived transitions, got {non_arrived_count}"


def test_replay_buffer_sample_stratified_fallback():
    """
    Test graceful fallback when buffer has fewer arrived transitions than requested,
    or when zero arrived transitions exist in the buffer.
    """
    capacity = 10
    state_dim = 2
    action_dim = 2
    buf = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=capacity)
    rng = np.random.default_rng(456)

    # Case A: 0 arrived transitions in buffer
    for i in range(8):
        buf.add(
            state=np.zeros(state_dim, dtype=np.float32),
            action=np.zeros(action_dim, dtype=np.float32),
            reward=float(i),
            next_state=np.zeros(state_dim, dtype=np.float32),
            done=False,
            arrived=False,
        )

    # Request 50% arrived when 0 exist -> should fall back to all available without error
    states, actions, rewards, next_states, dones = buf.sample_stratified(
        batch_size=4,
        arrived_fraction=0.5,
        rng=rng,
    )
    assert states.shape == (4, state_dim)
    assert np.all(rewards < 100.0)

    # Case B: Only 1 arrived transition exists, but batch requests 3 (round(4 * 0.75))
    buf.add(
        state=np.ones(state_dim, dtype=np.float32),
        action=np.ones(action_dim, dtype=np.float32),
        reward=999.0,
        next_state=np.ones(state_dim, dtype=np.float32),
        done=True,
        arrived=True,
    )
    states, actions, rewards, next_states, dones = buf.sample_stratified(
        batch_size=4,
        arrived_fraction=0.75,
        rng=rng,
    )
    assert states.shape == (4, state_dim)
    # The 1 arrived transition should be included, rest filled from non-arrived
    assert np.sum(rewards == 999.0) >= 1

