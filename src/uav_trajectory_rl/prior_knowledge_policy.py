"""
Prior-Knowledge Exploration Policy for PKTD3-TD.

This module implements the action selection mechanism corresponding to
equations (30)-(31) of the IEEE TNSE paper:
    "3-D Trajectory Design Based on Deep Reinforcement Learning for
     UAV-Assisted Communication Networks"

It provides:
    1. generate_prior_knowledge_action (eq. 30):
       Draws a heuristic prior-knowledge action a_n^pk that guides the UAV
       towards the destination horizontally without altitude fluctuation.
    2. unnormalize_action:
       Affine mapping from the actor network's normalized output space [-c, c]^3
       to the physical action space (v, lam, rho).
       DESIGN DECISION: The paper defines clipping in [-c, c] but does not
       specify the un-normalization mapping. We implement explicit affine scaling:
           v   = (raw[0] + c) / (2c) * V_MAX     # [-c, c] -> [0, V_MAX]
           lam = (raw[1] + c) / (2c) * pi         # [-c, c] -> [0, pi]
           rho = raw[2] * (pi / c)                # [-c, c] -> [-pi, pi]
    3. select_action (eq. 31):
       Dispatches between prior-knowledge action (when replay buffer size <= R_RAND)
       and noisy clipped actor network output (when replay buffer size > R_RAND).
"""

import math
from typing import Callable, Tuple
import numpy as np

from uav_trajectory_rl.config import (
    ACTION_CLIP_C,
    ANNEAL_STEPS,
    LAMBDA_PK,
    RHO_PK,
    R_RAND,
    SIGMA3,
    V_MAX,
)


def generate_prior_knowledge_action(rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Generate a prior-knowledge action a_n^pk (eq. 30).

    Formula:
        a_n^pk = { rand(0, 1) * V_MAX, LAMBDA_PK, rand(0, 1) * RHO_PK }

    Parameters:
        rng: NumPy random number generator instance.

    Returns:
        Tuple[float, float, float]: Physical action (v, lam, rho):
            - v: Flight speed in [0, V_MAX] (m/s).
            - lam: Polar angle fixed to LAMBDA_PK (0.5 * pi rad, horizontal flight).
            - rho: Azimuth angle drawn uniformly from [0, RHO_PK] (0 to 0.5 * pi rad).
    """
    v = float(rng.uniform(0.0, 1.0) * V_MAX)
    lam = float(LAMBDA_PK)
    rho = float(rng.uniform(0.0, 1.0) * RHO_PK)
    return (v, lam, rho)


def unnormalize_action(
    raw_action: np.ndarray,
    c: float = ACTION_CLIP_C,
) -> Tuple[float, float, float]:
    """
    Map actor network's normalized output space [-c, c]^3 to physical action ranges.

    DESIGN DECISION (not specified in the paper):
        The actor output space after tanh activation and clipping is [-c, c]^3.
        Physical action bounds are v in [0, V_MAX], lam in [0, pi], rho in [-pi, pi].
        We apply an explicit affine scaling:
            v   = (raw[0] + c) / (2.0 * c) * V_MAX
            lam = (raw[1] + c) / (2.0 * c) * math.pi
            rho = raw[2] * (math.pi / c)

    Parameters:
        raw_action: 3-element array-like in [-c, c]^3.
        c: Action clipping bound magnitude (default from config: ACTION_CLIP_C = 1.0).

    Returns:
        Tuple[float, float, float]: Un-normalized physical action (v, lam, rho).
    """
    v_raw, lam_raw, rho_raw = raw_action[0], raw_action[1], raw_action[2]
    v = (v_raw + c) / (2.0 * c) * V_MAX
    lam = (lam_raw + c) / (2.0 * c) * math.pi
    rho = rho_raw * (math.pi / c)
    return (float(v), float(lam), float(rho))


def normalize_action(
    physical_action: Tuple[float, float, float] | np.ndarray,
    c: float = ACTION_CLIP_C,
) -> np.ndarray:
    """
    Inverse of unnormalize_action: maps physical (v, lam, rho) back to the
    actor's normalized [-c, c]^3 output space. Exact algebraic inverse of the
    affine mapping in unnormalize_action -- must round-trip exactly:
        v_raw   = v / V_MAX * (2.0 * c) - c
        lam_raw = lam / math.pi * (2.0 * c) - c
        rho_raw = rho * (c / math.pi)

    Parameters:
        physical_action: 3-element tuple or array of physical (v, lam, rho).
        c: Action clipping bound magnitude (default from config: ACTION_CLIP_C = 1.0).

    Returns:
        np.ndarray: Normalized action vector of shape (3,) in [-c, c]^3, dtype float64.
    """
    v, lam, rho = physical_action[0], physical_action[1], physical_action[2]
    v_raw = (v / V_MAX) * (2.0 * c) - c
    lam_raw = (lam / math.pi) * (2.0 * c) - c
    rho_raw = rho * (c / math.pi)
    return np.array([v_raw, lam_raw, rho_raw], dtype=np.float64)


def select_action(
    state: np.ndarray,
    replay_buffer_size: int,
    actor_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    sigma3: float = SIGMA3,
    c: float = ACTION_CLIP_C,
    r_rand: int = R_RAND,
    anneal_steps: int = ANNEAL_STEPS,
) -> Tuple[float, float, float]:
    """
    Select an action according to the prior-knowledge exploration policy (eq. 31),
    with optional probabilistic annealing over transition window [R_rand, R_rand + anneal_steps].

    Formula:
        If R_ex <= R_rand:
            a_n = a_n^pk (pure prior knowledge)
        If R_rand < R_ex <= R_rand + anneal_steps:
            p_pk = 1.0 - (R_ex - R_rand) / anneal_steps
            a_n = a_n^pk with probability p_pk, else clip(actor_fn(s_n) + eps, -c, c)
        If R_ex > R_rand + anneal_steps:
            a_n = clip(actor_fn(s_n) + eps, -c, c) (pure network)

    Parameters:
        state: Current environment state vector s_n.
        replay_buffer_size: Current number of transition tuples in replay buffer (R_ex).
        actor_fn: Callable mapping state s_n -> np.ndarray of shape (3,) with values in [-c, c].
        rng: NumPy random number generator instance.
        sigma3: Standard deviation of exploration noise (default from config: SIGMA3 = 0.1).
        c: Action clipping bound (default from config: ACTION_CLIP_C = 1.0).
        r_rand: Number of pure prior-knowledge exploration steps (default: R_RAND = 20000).
        anneal_steps: Transition steps over which PK reliance linearly anneals from 1 to 0.
            If 0, performs an abrupt hard switch at R_rand (literal paper eq. 31).

    Returns:
        Tuple[float, float, float]: Physical action (v, lam, rho) ready for env.step().
    """
    if replay_buffer_size <= r_rand:
        return generate_prior_knowledge_action(rng)

    if anneal_steps > 0 and replay_buffer_size <= r_rand + anneal_steps:
        p_pk = 1.0 - float(replay_buffer_size - r_rand) / float(anneal_steps)
        if rng.random() < p_pk:
            return generate_prior_knowledge_action(rng)

    raw = np.asarray(actor_fn(state), dtype=np.float64)
    eps = rng.normal(0.0, sigma3, size=3)
    noisy = np.clip(raw + eps, -c, c)
    return unnormalize_action(noisy, c)
