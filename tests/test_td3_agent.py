import math
import numpy as np
import pytest
import torch

from uav_trajectory_rl.config import ACTION_CLIP_C, POLICY_DELAY
from uav_trajectory_rl.td3_agent import TD3Agent
from uav_trajectory_rl.td3_networks import ReplayBuffer


def test_td3_agent_construction():
    state_dim = 26
    action_dim = 3
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim)

    # Confirm actor and actor_target start with identical parameters
    actor_sd = agent.actor.state_dict()
    actor_target_sd = agent.actor_target.state_dict()
    assert set(actor_sd.keys()) == set(actor_target_sd.keys())
    for k in actor_sd:
        assert torch.equal(actor_sd[k], actor_target_sd[k]), f"Mismatch in actor param {k}"

    # Confirm critic and critic_target start with identical parameters
    critic_sd = agent.critic.state_dict()
    critic_target_sd = agent.critic_target.state_dict()
    assert set(critic_sd.keys()) == set(critic_target_sd.keys())
    for k in critic_sd:
        assert torch.equal(critic_sd[k], critic_target_sd[k]), f"Mismatch in critic param {k}"

    assert agent.total_updates == 0


def test_select_action():
    state_dim = 26
    action_dim = 3
    c = ACTION_CLIP_C
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, max_action=c)

    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state)

    assert isinstance(action, np.ndarray)
    assert action.shape == (action_dim,)
    assert (action >= -c).all() and (action <= c).all()


def test_train_step_and_delayed_update():
    state_dim = 26
    action_dim = 3
    batch_size = 32
    capacity = 100
    rng = np.random.default_rng(42)

    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, policy_delay=POLICY_DELAY)
    buf = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=capacity)

    # Fill buffer with 50 transitions
    for i in range(50):
        s = rng.normal(size=state_dim).astype(np.float32)
        a = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        r = float(rng.uniform(-1.0, 5.0))
        s_next = rng.normal(size=state_dim).astype(np.float32)
        d = (i % 15 == 0)  # occasional terminal transitions
        buf.add(s, a, r, s_next, d)

    num_steps = POLICY_DELAY * 3  # 6 steps

    # Save target parameters before first step
    initial_actor_target_weight = agent.actor_target.out_layer.weight.clone()
    initial_critic_target_weight = agent.critic_target.q1_out.weight.clone()

    for step in range(1, num_steps + 1):
        prev_actor_target_weight = agent.actor_target.out_layer.weight.clone()

        diag = agent.train_step(buf, batch_size=batch_size, rng=rng)

        # Critic loss must be a finite float
        assert isinstance(diag["critic_loss"], float)
        assert not math.isnan(diag["critic_loss"]) and not math.isinf(diag["critic_loss"])
        assert diag["total_updates"] == step

        if step % POLICY_DELAY == 0:
            # Delayed update happened: actor loss must be a float
            assert isinstance(diag["actor_loss"], float)
            assert not math.isnan(diag["actor_loss"]) and not math.isinf(diag["actor_loss"])
            # Target networks should have updated
            assert not torch.equal(agent.actor_target.out_layer.weight, prev_actor_target_weight)
        else:
            # No delayed update
            assert diag["actor_loss"] is None
            # Target networks should NOT have updated
            assert torch.equal(agent.actor_target.out_layer.weight, prev_actor_target_weight)

    # Target parameters after 6 steps must differ from initial values
    assert not torch.equal(agent.actor_target.out_layer.weight, initial_actor_target_weight)
    assert not torch.equal(agent.critic_target.q1_out.weight, initial_critic_target_weight)


def test_terminal_transition_zeroing():
    torch.manual_seed(42)
    state_dim = 26
    action_dim = 3
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim)

    # Explicitly bias critic targets so Q-value is guaranteed non-zero (> 1.0)
    with torch.no_grad():
        agent.critic_target.q1_out.bias.fill_(1.0)
        agent.critic_target.q2_out.bias.fill_(1.0)

    # 1. Terminal transition (done = 1.0)
    reward_val = 7.5
    rewards_term = torch.tensor([[reward_val]], dtype=torch.float32, device=agent.device)
    next_states = torch.randn(1, state_dim, dtype=torch.float32, device=agent.device)
    dones_term = torch.tensor([[1.0]], dtype=torch.float32, device=agent.device)

    target_term = agent._compute_target(rewards_term, next_states, dones_term)
    # With done=1, gamma * (1 - 1) * min(Q1, Q2) = 0, so target == reward exactly
    assert torch.isclose(target_term, rewards_term).item()

    # 2. Non-terminal transition (done = 0.0)
    dones_non_term = torch.tensor([[0.0]], dtype=torch.float32, device=agent.device)
    target_non_term = agent._compute_target(rewards_term, next_states, dones_non_term)
    # Target should include the discounted bootstrapped Q-value
    assert not torch.isclose(target_non_term, rewards_term).item()


def test_save_load(tmp_path):
    state_dim = 26
    action_dim = 3
    agent1 = TD3Agent(state_dim=state_dim, action_dim=action_dim)
    agent1.total_updates = 42

    # Run one update or perturb parameters so weights differ from default init
    dummy_s = torch.randn(4, state_dim, device=agent1.device)
    dummy_a = torch.randn(4, action_dim, device=agent1.device)
    loss = agent1.critic.q1_forward(dummy_s, dummy_a).sum()
    loss.backward()
    agent1.critic_optimizer.step()

    save_path = str(tmp_path / "td3_agent.pt")
    agent1.save(save_path)

    # Fresh agent with different random initialization
    agent2 = TD3Agent(state_dim=state_dim, action_dim=action_dim)
    assert agent2.total_updates == 0
    # Weights should not match before loading
    assert not torch.equal(agent1.critic.q1_out.weight, agent2.critic.q1_out.weight)

    agent2.load(save_path)

    # After loading, all state dicts must match exactly
    for k in agent1.actor.state_dict():
        assert torch.equal(agent1.actor.state_dict()[k], agent2.actor.state_dict()[k])
    for k in agent1.critic.state_dict():
        assert torch.equal(agent1.critic.state_dict()[k], agent2.critic.state_dict()[k])
    for k in agent1.actor_target.state_dict():
        assert torch.equal(agent1.actor_target.state_dict()[k], agent2.actor_target.state_dict()[k])
    for k in agent1.critic_target.state_dict():
        assert torch.equal(agent1.critic_target.state_dict()[k], agent2.critic_target.state_dict()[k])

    assert agent2.total_updates == agent1.total_updates
