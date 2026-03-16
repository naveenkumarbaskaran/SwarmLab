"""Standard benchmark functions for optimization testing."""

from __future__ import annotations

import numpy as np
from typing import Protocol


class BenchmarkFunction(Protocol):
    """Protocol for benchmark functions."""
    name: str
    bounds: tuple[float, float]
    global_minimum: float

    def __call__(self, x: np.ndarray) -> float: ...


class Rastrigin:
    """Rastrigin function — many local minima, global min = 0 at origin."""
    name = "Rastrigin"
    bounds = (-5.12, 5.12)
    global_minimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class Ackley:
    """Ackley function — nearly flat outer region with deep hole at center."""
    name = "Ackley"
    bounds = (-5.0, 5.0)
    global_minimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


class Schwefel:
    """Schwefel function — global min far from local minima."""
    name = "Schwefel"
    bounds = (-500.0, 500.0)
    global_minimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Griewank:
    """Griewank function — many local minima with product term."""
    name = "Griewank"
    bounds = (-600.0, 600.0)
    global_minimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_sq - prod_cos + 1


class Sphere:
    """Sphere function — simplest unimodal benchmark."""
    name = "Sphere"
    bounds = (-5.12, 5.12)
    global_minimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(x**2))


class Rosenbrock:
    """Rosenbrock function — narrow curved valley (banana function)."""
    name = "Rosenbrock"
    bounds = (-5.0, 10.0)
    global_minimum = 0.0

    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


# Registry
BENCHMARKS: dict[str, type] = {
    "Rastrigin": Rastrigin,
    "Ackley": Ackley,
    "Schwefel": Schwefel,
    "Griewank": Griewank,
    "Sphere": Sphere,
    "Rosenbrock": Rosenbrock,
}
