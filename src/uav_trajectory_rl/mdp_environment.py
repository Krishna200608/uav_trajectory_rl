"""
MDP Environment Wrapper for PKTD3-TD UAV Trajectory Design.

This module implements the full Markov Decision Process (MDP) environment as
described in Section IV (eq. 17-29) and Algorithm 1 of the IEEE TNSE paper:
    "3-D Trajectory Design Based on Deep Reinforcement Learning for
     UAV-Assisted Communication Networks"

It ties together:
    - uav_kinematics.py: position updates and acceleration constraints (eq. 1-3)
    - user_mobility.py:  Gauss-Markov ground user mobility (eq. 4-7)
    - channel_model.py:  LoS/NLoS path loss and transmission rate (eq. 8-14)
    - energy_model.py:   rotary-wing UAV propulsion energy (eq. 15-16)

Documented corrections from the source paper:
    - eq. (26): d_near uses Q_END (destination), not Q_START as literally typeset.
    - eq. (29) vs Algorithm 1 Line 15: all six reward terms implemented (eq. 29).
    - ARRIVAL_THRESHOLD_M is an assumption not given in the paper.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from uav_trajectory_rl.config import (
    AC_MAX,
    ARRIVAL_THRESHOLD_M,
    C_AC,
    C_AR,
    C_EN,
    C_H,
    C_LACK,
    C_NEAR,
    C_NR,
    C_TH,
    DELTA,
    MAX_DISTANCE,
    N_SLOTS,
    Q_END,
    Q_START,
    T_MAX,
    V_MAX,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
)
from uav_trajectory_rl.uav_kinematics import (
    apply_acceleration_constraint,
    step_uav_position,
)
from uav_trajectory_rl.user_mobility import UserSwarm
from uav_trajectory_rl.channel_model import (
    los_probability,
    total_transmission_rate,
)
from uav_trajectory_rl.energy_model import energy_consumption


class UAVTrajectoryEnv:
    """
    Gym-like MDP environment for the PKTD3-TD UAV trajectory design problem.

    State (eq. 19): s_n = [x, y, z, x1, y1, ..., xK, yK, v, t_re, d_re]
        Flat np.ndarray of shape (2*K + 6,).
        Components are normalized to roughly [-1, 1] (coordinates) and [0, 1] (v, t_re, d_re)
        to prevent actor-critic pre-activation saturation (DESIGN DECISION).

    Action (eq. 20): a_n = (v_n, lam_n, rho_n)
        Raw physical values: speed in [0, V_MAX], polar in [0, pi], azimuth in [-pi, pi].

    Reward (eq. 21-29): sum of six terms r_n,1..r_n,6.
        All reward terms evaluate against raw physical internal attributes, not normalized states.
    """

    def __init__(
        self,
        k: int = 10,
        rng: Optional[np.random.Generator] = None,
        # Override config constants if needed for experiments
        q_start: Tuple[float, float, float] = Q_START,
        q_end: Tuple[float, float, float] = Q_END,
        v_max: float = V_MAX,
        ac_max: float = AC_MAX,
        delta: float = DELTA,
        n_slots: int = N_SLOTS,
        t_max: float = T_MAX,
        x_min: float = X_MIN,
        x_max: float = X_MAX,
        y_min: float = Y_MIN,
        y_max: float = Y_MAX,
        z_min: float = Z_MIN,
        z_max: float = Z_MAX,
        arrival_threshold: float = ARRIVAL_THRESHOLD_M,
        c_th: float = C_TH,
        c_en: float = C_EN,
        c_ar: float = C_AR,
        c_nr: float = C_NR,
        c_ac: float = C_AC,
        c_h: float = C_H,
        c_near: float = C_NEAR,
        c_lack: float = C_LACK,
    ) -> None:
        """
        Initialize the UAV trajectory MDP environment.

        Parameters:
            k: Number of ground users.
            rng: NumPy random generator for reproducibility. Defaults to unseeded.
            q_start: UAV starting position (x, y, z) in meters.
            q_end: UAV destination position (x, y, z) in meters.
            v_max: Maximum UAV speed (m/s).
            ac_max: Maximum acceleration magnitude (m/s^2).
            delta: Time slot duration (s).
            n_slots: Maximum number of time slots per episode.
            t_max: Total mission time budget (s).
            x_min, x_max, y_min, y_max: Horizontal service area bounds (m).
            z_min, z_max: Altitude bounds (m).
            arrival_threshold: Distance tolerance for destination arrival (m).
            c_th..c_lack: Reward weight coefficients.
        """
        self.k: int = k
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        # Environment geometry
        self.q_start: np.ndarray = np.array(q_start, dtype=np.float64)
        self.q_end: np.ndarray = np.array(q_end, dtype=np.float64)
        self.v_max: float = v_max
        self.ac_max: float = ac_max
        self.delta: float = delta
        self.n_slots: int = n_slots
        self.t_max: float = t_max
        self.x_min: float = x_min
        self.x_max: float = x_max
        self.y_min: float = y_min
        self.y_max: float = y_max
        self.z_min: float = z_min
        self.z_max: float = z_max
        self.arrival_threshold: float = arrival_threshold

        # Reward coefficients
        self.c_th: float = c_th
        self.c_en: float = c_en
        self.c_ar: float = c_ar
        self.c_nr: float = c_nr
        self.c_ac: float = c_ac
        self.c_h: float = c_h
        self.c_near: float = c_near
        self.c_lack: float = c_lack

        # State dimensions: [x, y, z, x1, y1, ..., xK, yK, v, t_re, d_re]
        self.state_dim: int = 2 * self.k + 6

        # Internal state (initialized by reset())
        self.uav_pos: np.ndarray = np.zeros(3, dtype=np.float64)
        self.uav_speed: float = 0.0
        self.step_count: int = 0
        self.prev_dist_to_end: float = 0.0  # ||q_{n-1} - Q_END|| for d_near calc
        self.user_swarm: Optional[UserSwarm] = None

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state (Algorithm 1, Lines 1-3).

        Returns:
            np.ndarray: Initial state s_0 of shape (2*K + 6,).
        """
        self.uav_pos = self.q_start.copy()
        self.uav_speed = 0.0
        self.step_count = 0

        # Initial distance to destination for d_near computation in the first step
        self.prev_dist_to_end = float(np.linalg.norm(self.uav_pos - self.q_end))

        # Fresh user swarm
        self.user_swarm = UserSwarm(
            k=self.k,
            area_bounds=(self.x_min, self.x_max, self.y_min, self.y_max),
            rng=self.rng,
        )

        return self._build_state()

    def step(
        self,
        action: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step (Algorithm 1, Lines 7-15).

        Parameters:
            action: Tuple (v_n_raw, lam_n, rho_n) — raw physical action values.
                v_n_raw: Commanded speed in [0, V_MAX] (m/s).
                lam_n: Polar angle in [0, pi] (radians).
                rho_n: Azimuth angle in [-pi, pi] (radians).

        Returns:
            Tuple of (next_state, reward, done, info):
                next_state: np.ndarray of shape (2*K + 6,).
                reward: Total scalar reward r_n (sum of six terms).
                done: True if episode has terminated.
                info: Dict with individual reward terms and diagnostic flags.
        """
        v_n_raw, lam_n, rho_n = action
        q_prev = self.uav_pos.copy()

        # ======================================================================
        # 1. Acceleration constraint (Algorithm 1, Line 7)
        # ======================================================================
        actual_v, accel_violated = apply_acceleration_constraint(
            v_prev=self.uav_speed,
            requested_v=v_n_raw,
            ac_max=self.ac_max,
            delta=self.delta,
        )

        # ======================================================================
        # 2. Compute candidate next position (eq. 1-3)
        # ======================================================================
        candidate_pos = step_uav_position(
            position=q_prev,
            v=actual_v,
            lam=lam_n,
            rho=rho_n,
            delta=self.delta,
        )

        # ======================================================================
        # 3. Spatial constraint checks (Algorithm 1, Lines 9-11)
        # ======================================================================
        # Check height violation (C9) — flag for r_n,6 penalty
        height_violated = (
            candidate_pos[2] < self.z_min or candidate_pos[2] > self.z_max
        )

        # Check xy boundary violation (C7/C8)
        xy_violated = (
            candidate_pos[0] < self.x_min or candidate_pos[0] > self.x_max
            or candidate_pos[1] < self.y_min or candidate_pos[1] > self.y_max
        )

        # If ANY spatial constraint is violated, cancel action: position stays
        if xy_violated or height_violated:
            self.uav_pos = q_prev.copy()
        else:
            self.uav_pos = candidate_pos

        # Update UAV speed to the acceleration-constrained value regardless
        self.uav_speed = actual_v

        # ======================================================================
        # 4. Advance ground users (Gauss-Markov mobility)
        # ======================================================================
        self.user_swarm.step()

        # ======================================================================
        # 5. Compute reward terms (eq. 21-29)
        # ======================================================================
        user_positions = self.user_swarm.get_positions()

        # Current distance to destination (for d_near and d_re)
        dist_to_end = float(np.linalg.norm(self.uav_pos - self.q_end))

        # Advance step counter
        self.step_count += 1

        # Determine terminal condition
        arrived = dist_to_end <= self.arrival_threshold
        done = (self.step_count >= self.n_slots) or arrived

        # Individual reward terms
        r1 = self._reward_throughput(user_positions)
        r2 = self._reward_energy(actual_v, lam_n)
        r3 = self._reward_terminal(done, arrived, dist_to_end)
        r4 = self._reward_proximity(dist_to_end)
        r5 = self._reward_accel_penalty(accel_violated)
        r6 = self._reward_height_penalty(height_violated)

        total_reward = r1 + r2 + r3 + r4 + r5 + r6

        # Update prev_dist for next step's d_near computation
        self.prev_dist_to_end = dist_to_end

        # ======================================================================
        # 6. Build next state and info dict
        # ======================================================================
        next_state = self._build_state()

        # Compute LoS probability for diagnostics (UAV-to-centroid of users)
        dx_mean = self.uav_pos[0] - float(np.mean(user_positions[:, 0]))
        dy_mean = self.uav_pos[1] - float(np.mean(user_positions[:, 1]))
        h_dist_mean = math.sqrt(dx_mean**2 + dy_mean**2)
        p_los_diag = los_probability(self.uav_pos[2], h_dist_mean)

        # Total rate for diagnostics
        user_pos_3d = np.column_stack([
            user_positions, np.zeros(self.k, dtype=np.float64)
        ])
        total_rate = total_transmission_rate(self.uav_pos, user_pos_3d)

        info: Dict[str, Any] = {
            "r1_throughput": r1,
            "r2_energy": r2,
            "r3_terminal": r3,
            "r4_proximity": r4,
            "r5_accel": r5,
            "r6_height": r6,
            "total_rate_bps": total_rate,
            "energy_j": energy_consumption(actual_v, lam_n, delta=self.delta),
            "los_probability": p_los_diag,
            "accel_violated": accel_violated,
            "height_violated": height_violated,
            "xy_violated": xy_violated,
            "position_cancelled": xy_violated or height_violated,
            "arrived": arrived,
            "dist_to_end": dist_to_end,
            "actual_speed": actual_v,
            "step": self.step_count,
        }

        return next_state, total_reward, done, info

    # ==========================================================================
    # State builder
    # ==========================================================================

    def _build_state(self) -> np.ndarray:
        """
        Construct the flat normalized state vector s_n (eq. 19).

        Layout: [x_norm, y_norm, z_norm, x1_norm, y1_norm, ..., xK_norm, yK_norm, v_norm, t_re_norm, d_re_norm]
        Shape: (2*K + 6,)

        DESIGN DECISION (CRITICAL FIX):
            Raw physical values span widely differing ranges (positions to 600m, altitude to 200m,
            time to 200s, distance to ~848m, speed to 20m/s). Unnormalized inputs cause premature
            saturation in bounded-output (tanh) actor-critic networks. We normalize positions to
            roughly [-1, 1] and scalar metrics to roughly [0, 1] using known physical bounds from config.py.
            Raw attributes are retained on self for physics and reward computation.
        """
        user_pos = self.user_swarm.get_positions()  # (K, 2)

        t_re = self.t_max - self.step_count * self.delta
        d_re = float(np.linalg.norm(self.uav_pos - self.q_end))

        # Position normalization to [-1, 1]
        x_norm = (self.uav_pos[0] - self.x_min) / (self.x_max - self.x_min) * 2.0 - 1.0
        y_norm = (self.uav_pos[1] - self.y_min) / (self.y_max - self.y_min) * 2.0 - 1.0
        z_norm = (self.uav_pos[2] - self.z_min) / (self.z_max - self.z_min) * 2.0 - 1.0

        user_norm = np.empty_like(user_pos)
        user_norm[:, 0] = (user_pos[:, 0] - self.x_min) / (self.x_max - self.x_min) * 2.0 - 1.0
        user_norm[:, 1] = (user_pos[:, 1] - self.y_min) / (self.y_max - self.y_min) * 2.0 - 1.0

        # Scalar normalization to [0, 1]
        v_norm = self.uav_speed / self.v_max
        t_re_norm = t_re / self.t_max
        d_re_norm = d_re / MAX_DISTANCE

        state = np.empty(self.state_dim, dtype=np.float64)
        state[0] = x_norm
        state[1] = y_norm
        state[2] = z_norm
        state[3:3 + 2 * self.k] = user_norm.flatten()
        state[3 + 2 * self.k] = v_norm
        state[4 + 2 * self.k] = t_re_norm
        state[5 + 2 * self.k] = d_re_norm

        return state

    # ==========================================================================
    # Reward components (eq. 21-28)
    # ==========================================================================

    def _reward_throughput(self, user_positions: np.ndarray) -> float:
        """
        r_n,1 = C_TH * R_n * DELTA  (eq. 21)

        R_n is the total transmission rate (bps) from channel_model.total_transmission_rate.
        User positions are (K, 2); we append z=0 for the 3D channel model call.
        """
        user_pos_3d = np.column_stack([
            user_positions, np.zeros(self.k, dtype=np.float64)
        ])
        r_n = total_transmission_rate(self.uav_pos, user_pos_3d)
        return self.c_th * r_n * self.delta

    def _reward_energy(self, actual_v: float, lam_n: float) -> float:
        """
        r_n,2 = -C_EN * E_n * DELTA  (eq. 22)

        NOTE ON DOUBLE-DELTA: E_n = energy_consumption(v, lam) already equals
        P(v, lam) * DELTA internally (eq. 16), so this multiplication by DELTA
        again is redundant for unit purposes (energy * time = ???). However,
        the paper's eq. (22) literally prints "r_{n,2} = -c_{en} * E_n * delta",
        so we implement it as written. Since DELTA=1.0 in this project's config,
        this is numerically inconsequential. If DELTA is ever changed, revisit
        whether the extra DELTA factor is intentional or a paper redundancy.
        """
        e_n = energy_consumption(actual_v, lam_n, delta=self.delta)
        # ASSUMPTION: energy is always charged based on the commanded (actual_v, lam_n)
        # regardless of whether the position update was cancelled by spatial constraints.
        # The UAV still expends propulsion effort attempting the move even if the
        # resulting position is discarded. The paper does not document this case.
        return -self.c_en * e_n * self.delta

    def _reward_terminal(
        self, done: bool, arrived: bool, dist_to_end: float,
    ) -> float:
        """
        r_n,3: terminal-only reward (eq. 23).

        On the terminal step (episode ends due to N_SLOTS reached OR early arrival):
            r_n,3 = -( (1 - arrived) * C_AR * (d_re / V_MAX) + C_NR )
        On every non-terminal step:
            r_n,3 = 0

        Note: even on successful arrival, r_n,3 = -(0 + C_NR) = -C_NR, matching
        the paper's literal parenthesization. The term is never fully zero.

        The condition "N = T/delta" in eq. (23) is interpreted as "this is the
        terminal step of the episode" (which can occur before N_SLOTS via early
        arrival per Algorithm 1's loop-termination rule).
        """
        if not done:
            return 0.0

        arrived_flag = 1.0 if arrived else 0.0
        return -((1.0 - arrived_flag) * self.c_ar * (dist_to_end / self.v_max) + self.c_nr)

    def _reward_proximity(self, dist_to_end: float) -> float:
        """
        r_n,4 = -t_lack * (1 - C_NEAR * d_near / V_MAX)  (eq. 24)

        t_lack = min( max(d_re / V_MAX, 0), C_LACK )      (eq. 25)

        d_near = ||q_{n-1} - Q_END|| - ||q_n - Q_END||    (eq. 26, CORRECTED)

        CORRECTION: The paper's typeset eq. (26) uses Q_START (q_s) instead of
        Q_END (q_e): "d_near = ||q_{n-1} - q_s|| - ||q_n - q_s||". This
        contradicts the paper's own prose description ("the decrease in distance
        from the UAV's current position to the destination") and contradicts the
        reward term's purpose (encouraging progress TOWARD destination). Implemented
        using Q_END as the prose describes. This is almost certainly a typo in
        the published paper.
        """
        # d_near: positive when UAV moved closer to destination this step
        d_near = self.prev_dist_to_end - dist_to_end

        # t_lack: clamped remaining-time proxy (eq. 25)
        t_lack = min(max(dist_to_end / self.v_max, 0.0), self.c_lack)

        return -t_lack * (1.0 - self.c_near * d_near / self.v_max)

    def _reward_accel_penalty(self, accel_violated: bool) -> float:
        """
        r_n,5 = -C_AC * accel_violated  (eq. 27)
        """
        return -self.c_ac if accel_violated else 0.0

    def _reward_height_penalty(self, height_violated: bool) -> float:
        """
        r_n,6 = -C_H * height_violated  (eq. 28)

        Charged when the CANDIDATE z-position would fall outside [Z_MIN, Z_MAX],
        even though the position move itself is cancelled (Algorithm 1 Line 9).
        The agent is penalized for attempting an invalid altitude move.
        """
        return -self.c_h if height_violated else 0.0

