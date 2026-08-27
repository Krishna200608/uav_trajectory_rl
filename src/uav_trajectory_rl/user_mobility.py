"""
Ground-User Gauss-Markov Mobility Model.

This module implements a 2D Gauss-Markov mobility model for mobile ground users (GUs).
In each discrete time slot, each user's speed and direction of movement depend on:
    1. Their previous speed and heading (memory component via parameter OMEGA)
    2. The group swarm mean speed and heading
    3. Independent Gaussian random perturbations (SIGMA1, SIGMA2)
"""

import math
from typing import Optional, Tuple
import numpy as np

from uav_trajectory_rl.config import DELTA, OMEGA, SIGMA1, SIGMA2


class UserSwarm:
    """
    Simulates a group of K mobile ground users moving under a Gauss-Markov model.

    Attributes:
        k (int): Number of ground users.
        bounds (tuple[float, float, float, float]): (x_min, x_max, y_min, y_max).
        rng (np.random.Generator): Random number generator instance for reproducibility.
        positions (np.ndarray): Array of shape (K, 2) storing [x, y] coordinates.
        velocities (np.ndarray): Array of shape (K,) storing speeds v_k.
        directions (np.ndarray): Array of shape (K,) storing movement angles theta_k in rad.
    """

    def __init__(
        self,
        k: int,
        area_bounds: Tuple[float, float, float, float],
        rng: Optional[np.random.Generator] = None,
        v_init_range: Tuple[float, float] = (0.5, 2.0),
        omega: float = OMEGA,
        sigma1: float = SIGMA1,
        sigma2: float = SIGMA2,
        delta: float = DELTA,
    ) -> None:
        """
        Initialize K users uniformly in area_bounds with initial velocities and directions.

        Parameters:
            k: Number of users.
            area_bounds: 4-tuple (x_min, x_max, y_min, y_max).
            rng: NumPy random generator instance. Defaults to np.random.default_rng() if None.
            v_init_range: (min_speed, max_speed) for uniform speed initialization in m/s.
            omega: Gauss-Markov memory / correlation coefficient in [0, 1].
            sigma1: Standard deviation for velocity Gaussian noise.
            sigma2: Standard deviation for direction Gaussian noise.
            delta: Time slot duration in seconds.
        """
        if k <= 0:
            raise ValueError(f"Number of users k must be positive, got {k}")

        self.k: int = k
        self.x_min, self.x_max, self.y_min, self.y_max = area_bounds
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self.v_init_range: Tuple[float, float] = v_init_range

        self.omega: float = omega
        self.sigma1: float = sigma1
        self.sigma2: float = sigma2
        self.delta: float = delta

        # Precompute Gauss-Markov random scale: sqrt(1 - omega^2)
        self.random_scale: float = math.sqrt(max(0.0, 1.0 - self.omega**2))

        # Initialize user state arrays
        self.positions: np.ndarray = np.empty((self.k, 2), dtype=np.float64)
        self.velocities: np.ndarray = np.empty((self.k,), dtype=np.float64)
        self.directions: np.ndarray = np.empty((self.k,), dtype=np.float64)

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reinitialize all user positions, velocities, and directions.

        Returns:
            np.ndarray: Initial positions array of shape (K, 2).
        """
        # Uniform random positions inside bounding box
        self.positions[:, 0] = self.rng.uniform(self.x_min, self.x_max, size=self.k)
        self.positions[:, 1] = self.rng.uniform(self.y_min, self.y_max, size=self.k)

        # Initial velocities and directions
        v_min, v_max = self.v_init_range
        self.velocities[:] = self.rng.uniform(v_min, v_max, size=self.k)
        self.directions[:] = self.rng.uniform(-math.pi, math.pi, size=self.k)

        return self.positions.copy()

    def step(self, clip_to_bounds: bool = True) -> np.ndarray:
        """
        Perform a single time-slot Gauss-Markov mobility step in-place.

        Update equations:
            v_{n+1}^k = omega * v_n^k + (1 - omega) * v_mean + sqrt(1 - omega^2) * psi^k
            theta_{n+1}^k = omega * theta_n^k + (1 - omega) * theta_mean + sqrt(1 - omega^2) * phi^k
            x_{n+1}^k = x_n^k + v_{n+1}^k * cos(theta_{n+1}^k) * delta
            y_{n+1}^k = y_n^k + v_{n+1}^k * sin(theta_{n+1}^k) * delta

        Parameters:
            clip_to_bounds: If True, keep users bounded inside the simulation area.

        Returns:
            np.ndarray: Updated positions array of shape (K, 2).
        """
        # Current group mean velocity and direction across all users
        v_mean = float(np.mean(self.velocities))
        theta_mean = float(np.mean(self.directions))

        # Independent Gaussian perturbations: psi ~ N(0, sigma1), phi ~ N(0, sigma2)
        psi = self.rng.normal(loc=0.0, scale=self.sigma1, size=self.k)
        phi = self.rng.normal(loc=0.0, scale=self.sigma2, size=self.k)

        # Update velocities and directions according to Gauss-Markov formulation
        next_velocities = (
            self.omega * self.velocities
            + (1.0 - self.omega) * v_mean
            + self.random_scale * psi
        )
        # Ensure velocities remain non-negative
        self.velocities[:] = np.maximum(next_velocities, 0.0)

        next_directions = (
            self.omega * self.directions
            + (1.0 - self.omega) * theta_mean
            + self.random_scale * phi
        )
        # Normalize directions to [-pi, pi]
        self.directions[:] = (next_directions + math.pi) % (2.0 * math.pi) - math.pi

        # Update 2D positions
        dx = self.velocities * np.cos(self.directions) * self.delta
        dy = self.velocities * np.sin(self.directions) * self.delta

        self.positions[:, 0] += dx
        self.positions[:, 1] += dy

        if clip_to_bounds:
            self.positions[:, 0] = np.clip(self.positions[:, 0], self.x_min, self.x_max)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.y_min, self.y_max)

        return self.positions.copy()

    def get_positions(self) -> np.ndarray:
        """
        Return the current 2D positions of all K users.

        Returns:
            np.ndarray: Array of shape (K, 2) containing [x_k, y_k] for all k.
        """
        return self.positions.copy()

    def get_velocities(self) -> np.ndarray:
        """Return current user speeds array of shape (K,)."""
        return self.velocities.copy()

    def get_directions(self) -> np.ndarray:
        """Return current user directions array of shape (K,)."""
        return self.directions.copy()

