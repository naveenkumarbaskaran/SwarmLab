"""Hybrid PSO-Firefly Algorithm."""

from __future__ import annotations

import numpy as np
from typing import Callable
from swarmlab.algorithms.base import SwarmOptimizer


class HybridPSOFirefly(SwarmOptimizer):
    """
    Hybrid PSO-Firefly Algorithm.

    Combines PSO's fast convergence with Firefly's multi-modal search:
    - Early iterations: PSO-dominated (fast approach to promising regions)
    - Late iterations: Firefly-dominated (fine-grained local search)

    Parameters:
        pso_weight: Initial PSO influence (0-1)
        firefly_weight: Initial Firefly influence (0-1)
        switch_iteration: Iteration where influence balance shifts
        w: PSO inertia weight
        c1, c2: PSO cognitive/social coefficients
        alpha: Firefly randomization
        beta0: Firefly base attractiveness
        gamma: Firefly absorption
    """

    def __init__(
        self,
        n_particles: int = 60,
        dimensions: int = 30,
        bounds: tuple[float, float] = (-5.12, 5.12),
        max_iterations: int = 1000,
        pso_weight: float = 0.7,
        firefly_weight: float = 0.3,
        switch_iteration: int = 500,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
        alpha: float = 0.2,
        beta0: float = 1.0,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(n_particles, dimensions, bounds, max_iterations, seed)
        self.pso_weight = pso_weight
        self.firefly_weight = firefly_weight
        self.switch_iteration = switch_iteration
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

        # PSO state
        self.velocities: np.ndarray = np.zeros((n_particles, dimensions))
        self.personal_best_pos: np.ndarray = np.zeros((n_particles, dimensions))
        self.personal_best_fit: np.ndarray = np.full(n_particles, np.inf)
        self._iteration: int = 0

    def initialize(self) -> None:
        super().initialize()
        low, high = self.bounds
        v_range = (high - low) * 0.1
        self.velocities = self.rng.uniform(-v_range, v_range, (self.n_particles, self.dimensions))
        self.personal_best_pos = self.positions.copy()
        self._iteration = 0

    def step(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        """One hybrid iteration: blend PSO velocity with Firefly attraction."""
        self._iteration += 1

        # Dynamic weight balance
        if self._iteration < self.switch_iteration:
            pw = self.pso_weight
            fw = self.firefly_weight
        else:
            pw = self.firefly_weight  # swap: Firefly dominates
            fw = self.pso_weight

        # PSO component
        r1 = self.rng.random((self.n_particles, self.dimensions))
        r2 = self.rng.random((self.n_particles, self.dimensions))
        cognitive = self.c1 * r1 * (self.personal_best_pos - self.positions)
        social = self.c2 * r2 * (self.best_position - self.positions)
        pso_velocity = self.w * self.velocities + cognitive + social

        # Firefly component (top-k neighbours for efficiency)
        firefly_delta = np.zeros_like(self.positions)
        sorted_idx = np.argsort(self.fitness)
        k = min(5, self.n_particles)  # attract toward top-5

        for i in range(self.n_particles):
            for j in sorted_idx[:k]:
                if self.fitness[j] < self.fitness[i]:
                    r = np.linalg.norm(self.positions[i] - self.positions[j])
                    beta = self.beta0 * np.exp(-self.gamma * r**2)
                    firefly_delta[i] += beta * (self.positions[j] - self.positions[i])
            firefly_delta[i] += self.alpha * (self.rng.random(self.dimensions) - 0.5)

        # Blend
        self.velocities = pw * pso_velocity + fw * firefly_delta
        self.positions += self.velocities
        self.clip_bounds()

        # Evaluate
        for i in range(self.n_particles):
            fit = fitness_fn(self.positions[i])
            self.evaluations += 1
            self.fitness[i] = fit

            if fit < self.personal_best_fit[i]:
                self.personal_best_fit[i] = fit
                self.personal_best_pos[i] = self.positions[i].copy()

            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_position = self.positions[i].copy()
