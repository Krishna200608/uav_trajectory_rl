"""
Baselines subpackage for UAV trajectory optimization algorithms.

Contains benchmark algorithms evaluated against PKTD3-TD:
    - TDPK (M10): Trajectory Design based on Prior Knowledge
    - Dueling DQL (M11): Discrete deep Q-learning baseline
    - PPO (M12): Proximal Policy Optimization continuous baseline
    - Greedy (M13): Myopic immediate-throughput heuristic
"""

from uav_trajectory_rl.baselines.dueling_dql import (
    DiscreteReplayBuffer,
    DuelingDQLAgent,
    DuelingQNetwork,
    discrete_action_to_physical,
    physical_to_nearest_discrete_idx,
)
from uav_trajectory_rl.baselines.greedy import greedy_action, run_greedy_episode
from uav_trajectory_rl.baselines.ppo import (
    PPOActor,
    PPOAgent,
    PPOCritic,
    RolloutBuffer,
)
from uav_trajectory_rl.baselines.tdpk import run_tdpk_episode, tdpk_action

__all__ = [
    "tdpk_action",
    "run_tdpk_episode",
    "discrete_action_to_physical",
    "physical_to_nearest_discrete_idx",
    "DuelingQNetwork",
    "DiscreteReplayBuffer",
    "DuelingDQLAgent",
    "PPOActor",
    "PPOCritic",
    "RolloutBuffer",
    "PPOAgent",
    "greedy_action",
    "run_greedy_episode",
]

