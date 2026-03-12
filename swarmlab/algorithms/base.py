"""Base class for all swarm optimization algorithms."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: list[float]
    iterations: int
    evaluations: int
    algorithm: str

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "best_fitness": self.best_fitness,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
        }


class SwarmOptimizer(ABC):
    """Abstract base class for swarm optimizers."""

    def __init__(
        self,
        n_particles: int = 50,
        dimensions: int = 30,
        bounds: tuple[float, float] = (-5.12, 5.12),
        max_iterations: int = 1000,
        seed: int | None = None,
    ):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)

        # State
        self.positions: np.ndarray = np.zeros((n_particles, dimensions))
        self.fitness: np.ndarray = np.full(n_particles, np.inf)
        self.best_position: np.ndarray = np.zeros(dimensions)
        self.best_fitness: float = np.inf
        self.convergence: list[float] = []
        self.evaluations: int = 0

    def initialize(self) -> None:
        """Initialize particle positions uniformly within bounds."""
        low, high = self.bounds
        self.positions = self.rng.uniform(low, high, (self.n_particles, self.dimensions))

    def clip_bounds(self) -> None:
        """Clip positions to stay within bounds."""
        low, high = self.bounds
        np.clip(self.positions, low, high, out=self.positions)

    @abstractmethod
    def step(self, fitness_fn: Callable[[np.ndarray], float]) -> None:
        """Perform one iteration of the algorithm."""
        ...

    def optimize(self, fitness_fn: Callable[[np.ndarray], float]) -> OptimizationResult:
        """Run the full optimization loop."""
        self.initialize()

        # Evaluate initial positions
        for i in range(self.n_particles):
            fit = fitness_fn(self.positions[i])
            self.fitness[i] = fit
            self.evaluations += 1
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_position = self.positions[i].copy()

        # Main loop
        for _ in range(self.max_iterations):
            self.step(fitness_fn)
            self.convergence.append(self.best_fitness)

        return OptimizationResult(
            best_position=self.best_position,
            best_fitness=self.best_fitness,
            convergence_curve=self.convergence,
            iterations=self.max_iterations,
            evaluations=self.evaluations,
            algorithm=self.__class__.__name__,
        )
