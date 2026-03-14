"""Firefly Algorithm."""

from __future__ import annotations

import numpy as np
from typing import Callable
from swarmlab.algorithms.base import SwarmOptimizer


class FireflyAlgorithm(SwarmOptimizer):
    """
    Firefly Algorithm (Yang, 2008).

    Fireflies move toward brighter (fitter) neighbours with attraction
    that decreases with distance (light absorption).

    Parameters:
        alpha: Randomization parameter (step size)
        beta0: Attractiveness at distance 0
        gamma: Light absorption coefficient
    """

    def __init__(
        self,
        n_particles: int = 50,
        dimensions: int = 30,
        bounds: tuple[float, float] = (-5.12, 5.12),
        max_iterations: int = 1000,
        alpha: float = 0.2,
        beta0: float = 1.0,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(n_particles, dimensions, bounds, max_iterations, seed)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def step(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        """One Firefly iteration: pairwise attraction + random walk."""
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if self.fitness[j] < self.fitness[i]:  # j is brighter
                    # Distance
                    r = np.linalg.norm(self.positions[i] - self.positions[j])
                    # Attractiveness
                    beta = self.beta0 * np.exp(-self.gamma * r**2)
                    # Move i toward j
                    self.positions[i] += beta * (self.positions[j] - self.positions[i])
                    # Random perturbation
                    self.positions[i] += self.alpha * (self.rng.random(self.dimensions) - 0.5)

        self.clip_bounds()

        # Evaluate all
        for i in range(self.n_particles):
            fit = fitness_fn(self.positions[i])
            self.evaluations += 1
            self.fitness[i] = fit
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_position = self.positions[i].copy()
