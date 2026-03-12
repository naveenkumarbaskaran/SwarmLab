"""Particle Swarm Optimization (PSO)."""

from __future__ import annotations

import numpy as np
from typing import Callable
from swarmlab.algorithms.base import SwarmOptimizer


class PSO(SwarmOptimizer):
    """
    Standard Particle Swarm Optimization.

    Parameters:
        w: Inertia weight (default: 0.729, Clerc's constriction)
        c1: Cognitive coefficient (personal best attraction)
        c2: Social coefficient (global best attraction)
    """

    def __init__(
        self,
        n_particles: int = 50,
        dimensions: int = 30,
        bounds: tuple[float, float] = (-5.12, 5.12),
        max_iterations: int = 1000,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
        seed: int | None = None,
    ):
        super().__init__(n_particles, dimensions, bounds, max_iterations, seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # PSO-specific state
        self.velocities: np.ndarray = np.zeros((n_particles, dimensions))
        self.personal_best_pos: np.ndarray = np.zeros((n_particles, dimensions))
        self.personal_best_fit: np.ndarray = np.full(n_particles, np.inf)

    def initialize(self) -> None:
        super().initialize()
        low, high = self.bounds
        v_range = (high - low) * 0.1
        self.velocities = self.rng.uniform(-v_range, v_range, (self.n_particles, self.dimensions))
        self.personal_best_pos = self.positions.copy()

    def step(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        """One PSO iteration: update velocity → position → evaluate."""
        r1 = self.rng.random((self.n_particles, self.dimensions))
        r2 = self.rng.random((self.n_particles, self.dimensions))

        # Velocity update
        cognitive = self.c1 * r1 * (self.personal_best_pos - self.positions)
        social = self.c2 * r2 * (self.best_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social

        # Position update
        self.positions += self.velocities
        self.clip_bounds()

        # Evaluate
        for i in range(self.n_particles):
            fit = fitness_fn(self.positions[i])
            self.evaluations += 1

            if fit < self.personal_best_fit[i]:
                self.personal_best_fit[i] = fit
                self.personal_best_pos[i] = self.positions[i].copy()

            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_position = self.positions[i].copy()
