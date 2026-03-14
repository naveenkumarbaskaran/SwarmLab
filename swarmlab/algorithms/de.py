"""Differential Evolution (DE/rand/1/bin)."""

from __future__ import annotations

import numpy as np
from typing import Callable
from swarmlab.algorithms.base import SwarmOptimizer


class DifferentialEvolution(SwarmOptimizer):
    """
    Differential Evolution (Storn & Price, 1997).

    Strategy: DE/rand/1/bin

    Parameters:
        F: Mutation factor (differential weight)
        CR: Crossover probability
    """

    def __init__(
        self,
        n_particles: int = 50,
        dimensions: int = 30,
        bounds: tuple[float, float] = (-5.12, 5.12),
        max_iterations: int = 1000,
        F: float = 0.8,
        CR: float = 0.9,
        seed: int | None = None,
    ):
        super().__init__(n_particles, dimensions, bounds, max_iterations, seed)
        self.F = F
        self.CR = CR

    def step(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        """One DE iteration: mutation → crossover → selection."""
        for i in range(self.n_particles):
            # Select 3 random distinct individuals (not i)
            candidates = [j for j in range(self.n_particles) if j != i]
            a, b, c = self.rng.choice(candidates, 3, replace=False)

            # Mutation: v = x_a + F * (x_b - x_c)
            mutant = self.positions[a] + self.F * (self.positions[b] - self.positions[c])

            # Crossover (binomial)
            trial = self.positions[i].copy()
            j_rand = self.rng.integers(self.dimensions)
            for j in range(self.dimensions):
                if self.rng.random() < self.CR or j == j_rand:
                    trial[j] = mutant[j]

            # Bounds
            low, high = self.bounds
            trial = np.clip(trial, low, high)

            # Selection
            trial_fit = fitness_fn(trial)
            self.evaluations += 1

            if trial_fit <= self.fitness[i]:
                self.positions[i] = trial
                self.fitness[i] = trial_fit

                if trial_fit < self.best_fitness:
                    self.best_fitness = trial_fit
                    self.best_position = trial.copy()
