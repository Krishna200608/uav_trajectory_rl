"""
Shared Evaluation Harness for UAV Trajectory RL (Module M14-Core).

Reference:
    M. Li et al., "3-D Trajectory Design Based on Deep Reinforcement Learning for
    UAV-Assisted Communication Networks," IEEE TNSE, vol. 13, no. 1, pp. 248-261, 2026.

This module provides a unified, cached evaluation harness for all five methods:
    1. TDPK (M10): Pure geometric direct flight baseline
    2. Greedy (M13): One-step myopic candidate lookahead baseline
    3. Dueling DQL (M11): Discrete-action baseline (checkpoints/dueling_dql_run1/)
    4. PPO (M12): Continuous on-policy stochastic actor-critic baseline (checkpoints/ppo_run1/)
    5. PKTD3-TD (M9): Prior-knowledge TD3 reference run (checkpoints/run4/)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from uav_trajectory_rl.baselines.dueling_dql import (
    DuelingDQLAgent,
    discrete_action_to_physical,
)
from uav_trajectory_rl.baselines.greedy import greedy_action
from uav_trajectory_rl.baselines.ppo import PPOAgent
from uav_trajectory_rl.baselines.tdpk import tdpk_action
from uav_trajectory_rl.config import (
    NUM_DISCRETE_ACTIONS,
    Q_END,
    Q_START,
    V_MAX,
)
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import unnormalize_action
from uav_trajectory_rl.td3_agent import TD3Agent


# ==============================================================================
# Method Specification Dataclass
# ==============================================================================

@dataclass
class MethodSpec:
    """
    Specification for a trajectory planning method.

    Attributes:
        name: Identifier ("TDPK", "Greedy", "DuelingDQL", "PPO", "PKTD3-TD").
        action_fn: Callable mapping (state, env) -> physical action (v, lam, rho).
        requires_checkpoint: Whether this method depends on trained weights on disk.
    """
    name: str
    action_fn: Callable[[np.ndarray, UAVTrajectoryEnv], Tuple[float, float, float]]
    requires_checkpoint: bool


# ==============================================================================
# Episode Log Dataclass & Serialization
# ==============================================================================

@dataclass
class EpisodeLog:
    """
    Detailed log of a single simulated evaluation episode.

    Captures step-by-step physical trajectories, channel metrics, energy,
    and aggregate performance indicators.
    """
    method_name: str
    seed: int
    k: int
    user_v_init_range: Tuple[float, float]
    steps_taken: int
    arrived: bool

    # Step traces (arrays of length T = steps_taken)
    positions: np.ndarray              # shape (T, 3): UAV (x, y, z) after each step
    start_pos: np.ndarray              # shape (3,): UAV initial position
    target_pos: np.ndarray             # shape (3,): UAV destination position
    actions: np.ndarray                # shape (T, 3): Physical actions (v, lam, rho)
    rewards: np.ndarray                # shape (T,): Total MDP reward per step
    distances_to_end: np.ndarray       # shape (T,): Distance to Q_END per step (m)
    los_probabilities: np.ndarray      # shape (T,): LoS probability per step
    transmission_rates_bps: np.ndarray # shape (T,): Instantaneous total rate (bps)
    energy_consumptions_j: np.ndarray  # shape (T,): Energy consumed per step (J)
    position_cancelled: np.ndarray     # shape (T,): bool flag for boundary violation
    user_positions_history: np.ndarray # shape (T, k, 2): all users' (x, y) after each step
    arrived_flags: np.ndarray          # shape (T,): bool flag for arrival condition

    # Summary metrics
    total_reward: float
    total_energy: float
    mean_throughput: float
    mean_los_probability: float
    max_displacement: float
    min_dist_to_end: float
    final_dist_to_end: float
    cancellation_rate_last50: float
    cancellation_rate_total: float

    def save_npz(self, filepath: str | Path) -> None:
        """Save episode log to compressed .npz archive on disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            method_name=np.array(self.method_name),
            seed=np.array(self.seed),
            k=np.array(self.k),
            user_v_init_range=np.array(self.user_v_init_range, dtype=np.float64),
            steps_taken=np.array(self.steps_taken),
            arrived=np.array(self.arrived),
            positions=self.positions,
            start_pos=self.start_pos,
            target_pos=self.target_pos,
            actions=self.actions,
            rewards=self.rewards,
            distances_to_end=self.distances_to_end,
            los_probabilities=self.los_probabilities,
            transmission_rates_bps=self.transmission_rates_bps,
            energy_consumptions_j=self.energy_consumptions_j,
            position_cancelled=self.position_cancelled,
            user_positions_history=self.user_positions_history,
            arrived_flags=self.arrived_flags,
            total_reward=np.array(self.total_reward, dtype=np.float64),
            total_energy=np.array(self.total_energy, dtype=np.float64),
            mean_throughput=np.array(self.mean_throughput, dtype=np.float64),
            mean_los_probability=np.array(self.mean_los_probability, dtype=np.float64),
            max_displacement=np.array(self.max_displacement, dtype=np.float64),
            min_dist_to_end=np.array(self.min_dist_to_end, dtype=np.float64),
            final_dist_to_end=np.array(self.final_dist_to_end, dtype=np.float64),
            cancellation_rate_last50=np.array(self.cancellation_rate_last50, dtype=np.float64),
            cancellation_rate_total=np.array(self.cancellation_rate_total, dtype=np.float64),
        )

    @classmethod
    def from_npz(cls, filepath: str | Path) -> EpisodeLog:
        """Load episode log from compressed .npz archive."""
        with np.load(filepath, allow_pickle=False) as data:
            v_range = data["user_v_init_range"]
            return cls(
                method_name=str(data["method_name"]),
                seed=int(data["seed"]),
                k=int(data["k"]),
                user_v_init_range=(float(v_range[0]), float(v_range[1])),
                steps_taken=int(data["steps_taken"]),
                arrived=bool(data["arrived"]),
                positions=data["positions"],
                start_pos=data["start_pos"],
                target_pos=data["target_pos"],
                actions=data["actions"],
                rewards=data["rewards"],
                distances_to_end=data["distances_to_end"],
                los_probabilities=data["los_probabilities"],
                transmission_rates_bps=data["transmission_rates_bps"],
                energy_consumptions_j=data["energy_consumptions_j"],
                position_cancelled=data["position_cancelled"],
                user_positions_history=data["user_positions_history"],
                arrived_flags=data["arrived_flags"],
                total_reward=float(data["total_reward"]),
                total_energy=float(data["total_energy"]),
                mean_throughput=float(data["mean_throughput"]),
                mean_los_probability=float(data["mean_los_probability"]),
                max_displacement=float(data["max_displacement"]),
                min_dist_to_end=float(data["min_dist_to_end"]),
                final_dist_to_end=float(data["final_dist_to_end"]),
                cancellation_rate_last50=float(data["cancellation_rate_last50"]),
                cancellation_rate_total=float(data["cancellation_rate_total"]),
            )


# ==============================================================================
# Factory for Method Specs
# ==============================================================================

DEFAULT_CHECKPOINT_PATHS: Dict[str, str] = {
    "DuelingDQL": "checkpoints/dueling_dql_run1/dueling_dql_final.pt",
    "PPO": "checkpoints/ppo_run1/ppo_final.pt",
    "PKTD3-TD": "checkpoints/run4/td3_agent_final.pt",
}


def get_method_specs(
    device: str | torch.device = "cpu",
    k: int = 10,
    checkpoint_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, MethodSpec]:
    """
    Construct and return MethodSpec dictionary for all 5 evaluation methods.

    Parameters:
        device: PyTorch device for neural network inference (default: "cpu").
        k: Ground user count defining state dimension (default: 10, state_dim=26).
        checkpoint_paths: Optional override dictionary mapping method names to
            checkpoint file paths.

    Returns:
        Dict[str, MethodSpec]: Dictionary mapping method names to their executable specs.
    """
    ckpt_map = dict(DEFAULT_CHECKPOINT_PATHS)
    if checkpoint_paths is not None:
        ckpt_map.update(checkpoint_paths)

    state_dim = 2 * k + 6
    torch_device = torch.device(device) if isinstance(device, str) else device

    specs: Dict[str, MethodSpec] = {}

    # 1. TDPK Baseline (M10): Pure geometric direct-flight policy
    specs["TDPK"] = MethodSpec(
        name="TDPK",
        action_fn=lambda state, env: tdpk_action(env.uav_pos, env.q_end, env.v_max, env.rng),
        requires_checkpoint=False,
    )

    # 2. Greedy Baseline (M13): One-step immediate-reward lookahead
    specs["Greedy"] = MethodSpec(
        name="Greedy",
        action_fn=lambda state, env: greedy_action(env),
        requires_checkpoint=False,
    )

    # 3. Dueling DQL Baseline (M11): Discrete Q-learning
    dql_path = ckpt_map["DuelingDQL"]
    if not os.path.exists(dql_path):
        raise FileNotFoundError(f"Dueling DQL checkpoint not found at: {dql_path}")
    dql_agent = DuelingDQLAgent(
        state_dim=state_dim,
        num_actions=NUM_DISCRETE_ACTIONS,
        device=torch_device,
    )
    dql_agent.load(dql_path)
    dql_agent.q_net.eval()
    dql_agent.q_target.eval()

    def _dql_action(state: np.ndarray, env: UAVTrajectoryEnv) -> Tuple[float, float, float]:
        a_idx = dql_agent.select_action(state, epsilon=0.0)
        return discrete_action_to_physical(a_idx)

    specs["DuelingDQL"] = MethodSpec(
        name="DuelingDQL",
        action_fn=_dql_action,
        requires_checkpoint=True,
    )

    # 4. PPO Baseline (M12): Continuous Gaussian actor-critic
    ppo_path = ckpt_map["PPO"]
    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"PPO checkpoint not found at: {ppo_path}")
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        device=torch_device,
    )
    ppo_agent.load(ppo_path)
    ppo_agent.actor.eval()
    ppo_agent.critic.eval()

    def _ppo_action(state: np.ndarray, env: UAVTrajectoryEnv) -> Tuple[float, float, float]:
        # NOTE: select_action_deterministic() internally unnormalizes Gaussian mean
        # directly to physical (v, lam, rho). Do NOT call unnormalize_action again!
        return ppo_agent.select_action_deterministic(state)

    specs["PPO"] = MethodSpec(
        name="PPO",
        action_fn=_ppo_action,
        requires_checkpoint=True,
    )

    # 5. PKTD3-TD Reference (M9): Prior-knowledge TD3 run4 checkpoint
    td3_path = ckpt_map["PKTD3-TD"]
    if not os.path.exists(td3_path):
        raise FileNotFoundError(f"PKTD3-TD checkpoint not found at: {td3_path}")
    td3_agent = TD3Agent(
        state_dim=state_dim,
        device=torch_device,
    )
    td3_agent.load(td3_path)
    td3_agent.actor.eval()

    def _td3_action(state: np.ndarray, env: UAVTrajectoryEnv) -> Tuple[float, float, float]:
        # TD3Agent.select_action() returns a normalized action in [-c, c]^3.
        # We MUST unnormalize via prior_knowledge_policy.unnormalize_action() to get physical action.
        raw_action = td3_agent.select_action(state)
        physical_act = unnormalize_action(raw_action)
        return physical_act

    specs["PKTD3-TD"] = MethodSpec(
        name="PKTD3-TD",
        action_fn=_td3_action,
        requires_checkpoint=True,
    )

    return specs


# ==============================================================================
# Simulation Execution Functions
# ==============================================================================

def run_episode(
    method: MethodSpec,
    seed: int,
    k: int = 10,
    user_v_init_range: Tuple[float, float] = (0.5, 2.0),
    max_steps: int = 200,
) -> EpisodeLog:
    """
    Execute one full deterministic evaluation episode for a given MethodSpec.

    Parameters:
        method: The MethodSpec to evaluate.
        seed: Random seed for environment initialization and mobility.
        k: Number of ground users.
        user_v_init_range: Speed range (min_v, max_v) in m/s for initial user velocities.
        max_steps: Maximum time slots per episode (default: 200).

    Returns:
        EpisodeLog: Complete step-by-step trace and aggregate metrics.
    """
    env = UAVTrajectoryEnv(
        k=k,
        rng=np.random.default_rng(seed),
        user_v_init_range=user_v_init_range,
        n_slots=max_steps,
    )
    state = env.reset()
    start_pos = env.uav_pos.copy()
    target_pos = env.q_end.copy()

    positions_list = []
    user_positions_list = []
    actions_list = []
    rewards_list = []
    distances_list = []
    los_list = []
    rates_list = []
    energy_list = []
    cancelled_list = []
    arrived_list = []

    done = False
    steps = 0
    arrived = False

    while not done and steps < max_steps:
        action = method.action_fn(state, env)
        next_state, reward, done, info = env.step(action)

        positions_list.append(env.uav_pos.copy())
        user_positions_list.append(env.user_swarm.positions.copy())
        actions_list.append(action)
        rewards_list.append(float(reward))
        distances_list.append(float(info["dist_to_end"]))
        los_list.append(float(info["los_probability"]))
        rates_list.append(float(info["total_rate_bps"]))
        energy_list.append(float(info["energy_j"]))
        cancelled_list.append(bool(info["position_cancelled"]))
        
        arr = bool(info["arrived"])
        arrived_list.append(arr)
        if arr:
            arrived = True

        state = next_state
        steps += 1

    positions = np.array(positions_list, dtype=np.float64)
    user_positions = np.array(user_positions_list, dtype=np.float64)
    actions = np.array(actions_list, dtype=np.float64)
    rewards = np.array(rewards_list, dtype=np.float64)
    distances = np.array(distances_list, dtype=np.float64)
    los_probs = np.array(los_list, dtype=np.float64)
    rates = np.array(rates_list, dtype=np.float64)
    energy = np.array(energy_list, dtype=np.float64)
    cancelled = np.array(cancelled_list, dtype=bool)
    arrived_flags = np.array(arrived_list, dtype=bool)

    if len(positions) > 0:
        displacements = np.linalg.norm(positions - start_pos, axis=1)
        max_disp = float(np.max(displacements))
        min_dist = float(np.min(distances))
        final_dist = float(distances[-1])
    else:
        init_dist = float(np.linalg.norm(start_pos - target_pos))
        max_disp = 0.0
        min_dist = init_dist
        final_dist = init_dist

    last50_slice = cancelled[-50:] if len(cancelled) >= 50 else cancelled
    canc_last50 = float(np.mean(last50_slice)) if len(last50_slice) > 0 else 0.0
    canc_total = float(np.mean(cancelled)) if len(cancelled) > 0 else 0.0

    return EpisodeLog(
        method_name=method.name,
        seed=seed,
        k=k,
        user_v_init_range=user_v_init_range,
        steps_taken=steps,
        arrived=arrived,
        positions=positions,
        start_pos=start_pos,
        target_pos=target_pos,
        actions=actions,
        rewards=rewards,
        distances_to_end=distances,
        los_probabilities=los_probs,
        transmission_rates_bps=rates,
        energy_consumptions_j=energy,
        position_cancelled=cancelled,
        user_positions_history=user_positions,
        arrived_flags=arrived_flags,
        total_reward=float(np.sum(rewards)),
        total_energy=float(np.sum(energy)),
        mean_throughput=float(np.mean(rates)) if len(rates) > 0 else 0.0,
        mean_los_probability=float(np.mean(los_probs)) if len(los_probs) > 0 else 0.0,
        max_displacement=max_disp,
        min_dist_to_end=min_dist,
        final_dist_to_end=final_dist,
        cancellation_rate_last50=canc_last50,
        cancellation_rate_total=canc_total,
    )


def run_batch(
    method_name: str,
    seeds: Sequence[int],
    k: int = 10,
    user_v_init_range: Tuple[float, float] = (0.5, 2.0),
    cache_dir: str | Path = "results/m14_cache",
    force: bool = False,
    device: str | torch.device = "cpu",
    method_spec: Optional[MethodSpec] = None,
) -> list[EpisodeLog]:
    """
    Run evaluation across multiple seeds with on-disk caching.

    Parameters:
        method_name: Identifier of the method ("TDPK", "Greedy", "DuelingDQL", "PPO", "PKTD3-TD").
        seeds: Sequence of integer random seeds to simulate.
        k: Ground user count.
        user_v_init_range: Initial user speed range (min_v, max_v).
        cache_dir: Directory where per-seed .npz archives are stored.
        force: If True, ignore existing cache files and re-simulate.
        device: Computation device for neural network inference.
        method_spec: Optional pre-constructed MethodSpec (if None, loaded on demand).

    Returns:
        list[EpisodeLog]: Evaluated or cached EpisodeLog objects in seed order.
    """
    cache_path_dir = Path(cache_dir)
    cache_path_dir.mkdir(parents=True, exist_ok=True)

    spec = method_spec
    logs: list[EpisodeLog] = []

    v_lo, v_hi = user_v_init_range

    for seed in seeds:
        cache_file = cache_path_dir / f"{method_name}_k{k}_v{v_lo:.1f}-{v_hi:.1f}_seed{seed}.npz"
        
        if not force and cache_file.exists():
            try:
                log = EpisodeLog.from_npz(cache_file)
                logs.append(log)
                continue
            except Exception:
                # If corrupted, fall back to re-simulating
                pass

        if spec is None:
            specs = get_method_specs(device=device, k=k)
            if method_name not in specs:
                raise ValueError(f"Unknown method '{method_name}'. Available: {list(specs.keys())}")
            spec = specs[method_name]

        log = run_episode(
            method=spec,
            seed=seed,
            k=k,
            user_v_init_range=user_v_init_range,
        )
        log.save_npz(cache_file)
        logs.append(log)

    return logs
