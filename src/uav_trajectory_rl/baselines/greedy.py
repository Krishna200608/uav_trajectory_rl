"""
Greedy Baseline for 3D UAV Trajectory Design (M13).

The IEEE TNSE reference paper evaluates Greedy [83] as:
  "Greedy algorithm [83]: The greedy algorithm makes decisions based on the
   current system performance. At each time slot, it selects the action that
   maximizes the immediate objective function to pursue a local optimal
   solution under the current system state."

This is a non-learning, myopic heuristic: no neural network, no training loop,
no replay buffer. At each step it evaluates all candidate actions and picks
whichever yields the highest IMMEDIATE (single-step) reward, ignoring
all future consequences entirely.

Design decisions:
  1. Candidate action set (DESIGN DECISION -- not paper-specified):
     Reuses the same 5×5×8 = 200 discrete action grid defined for M11 (Dueling
     DQL) via V_LEVELS, LAM_LEVELS, RHO_LEVELS and discrete_action_to_physical
     in config.py / baselines/dueling_dql.py. This gives a consistent, finite,
     and comparable candidate pool across baselines, covering the full
     (speed, polar, azimuth) product space.

  2. One-step lookahead via deep copy (DESIGN DECISION):
     UAVTrajectoryEnv.step() mutates internal state. To evaluate candidate
     rewards without corrupting the real environment, each candidate action is
     evaluated by deep-copying the environment (copy.deepcopy), calling step()
     on the copy, reading the resulting reward, then discarding the copy. The
     real environment's state is NEVER modified during the search; only the
     chosen best action is applied via a single real env.step() call after the
     search completes.

     Trade-off: 200 deep copies + step() calls per real step makes this ~200x
     more expensive per step than a direct policy. This is intentional and
     acceptable for a non-training baseline evaluated once over 20 seeds.
     Measured wall-clock time is reported in docs/PKTD3-TD_Tracker.md.

  3. Tie-breaking: first-encountered action wins (standard for greedy scan).
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from uav_trajectory_rl.baselines.dueling_dql import discrete_action_to_physical
from uav_trajectory_rl.config import NUM_DISCRETE_ACTIONS, Q_START
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv


def greedy_action(env: UAVTrajectoryEnv) -> Tuple[float, float, float]:
    """
    Select the greedy action that maximizes the single-step immediate reward.

    Evaluation procedure (DESIGN DECISION: deep copy + step):
      For each of the 200 discrete candidate actions:
        1. Deep-copy the real environment (preserving its full internal state).
        2. Call copy.step(candidate_action) to get the immediate reward.
        3. Discard the copy.
      Return the candidate action with the highest immediate reward.

    The real ``env`` is NEVER modified during this search. Calling code is
    responsible for applying the returned action to ``env`` via a separate
    ``env.step()`` call.

    Tie-breaking: first-encountered maximum action is kept (standard scan).

    Args:
        env: UAVTrajectoryEnv with the current episode state.

    Returns:
        Tuple (v, lam, rho) physical action that maximizes the immediate reward.
    """
    best_action: Tuple[float, float, float] | None = None
    best_reward: float = -float("inf")

    for idx in range(NUM_DISCRETE_ACTIONS):
        v, lam, rho = discrete_action_to_physical(idx)

        # Simulate without touching the real env
        env_copy = copy.deepcopy(env)
        _, reward, _, _ = env_copy.step((v, lam, rho))

        if reward > best_reward:
            best_reward = reward
            best_action = (v, lam, rho)

    # best_action is guaranteed non-None because NUM_DISCRETE_ACTIONS >= 1
    assert best_action is not None, "Candidate set was empty -- check NUM_DISCRETE_ACTIONS"
    return best_action


def run_greedy_episode(env: UAVTrajectoryEnv) -> Dict[str, Any]:
    """
    Run one complete evaluation episode under the greedy one-step lookahead policy.

    Mirrors M10's run_tdpk_episode() interface exactly for direct comparability:
    same return dict keys, same trajectory shape convention.

    Args:
        env: UAVTrajectoryEnv instance (will be reset at start).

    Returns:
        Dict[str, Any]:
            - episode_reward: float  -- cumulative 6-term MDP reward
            - steps_taken:    int    -- number of env.step() calls made
            - arrived:        bool   -- True if UAV reached Q_END within threshold
            - trajectory:     np.ndarray of shape (steps_taken + 1, 3),
                              UAV 3D positions starting at Q_START (index 0)
    """
    env.reset()
    trajectory: List[np.ndarray] = [env.uav_pos.copy()]
    episode_reward = 0.0
    done = False
    steps = 0
    arrived = False

    while not done:
        action = greedy_action(env)
        _, reward, done, info = env.step(action)
        trajectory.append(env.uav_pos.copy())
        episode_reward += reward
        steps += 1
        arrived = info["arrived"]

    return {
        "episode_reward": float(episode_reward),
        "steps_taken": int(steps),
        "arrived": bool(arrived),
        "trajectory": np.array(trajectory, dtype=np.float64),
    }
