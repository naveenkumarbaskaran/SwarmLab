"""Ablation study runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarmlab.algorithms.base import OptimizationResult
from swarmlab.algorithms import PSO, FireflyAlgorithm, HybridPSOFirefly, DifferentialEvolution
from swarmlab.benchmarks.functions import BENCHMARKS


ALGORITHM_MAP = {
    "PSO": PSO,
    "Firefly": FireflyAlgorithm,
    "HybridPSOFirefly": HybridPSOFirefly,
    "DE": DifferentialEvolution,
}


@dataclass
class AblationResults:
    """Collected results from an ablation study."""
    records: list[dict[str, Any]] = field(default_factory=list)

    def summary_table(self) -> str:
        """Generate markdown table of mean results."""
        if not self.records:
            return "No results."

        lines = ["| Algorithm | Benchmark | Dims | Mean Fitness | Std | Runs |",
                 "|-----------|-----------|------|-------------|-----|------|"]
        # Group by (algo, bench, dims)
        from collections import defaultdict
        groups: dict[tuple, list[float]] = defaultdict(list)
        for r in self.records:
            key = (r["algorithm"], r["benchmark"], r["dimensions"])
            groups[key].append(r["best_fitness"])

        for (algo, bench, dims), fits in sorted(groups.items()):
            mean = np.mean(fits)
            std = np.std(fits)
            lines.append(f"| {algo} | {bench} | {dims} | {mean:.4f} | {std:.4f} | {len(fits)} |")

        return "\n".join(lines)

    def to_latex(self, filepath: str) -> None:
        """Export as LaTeX table."""
        # Simplified — real implementation writes full .tex
        with open(filepath, "w") as f:
            f.write(self.summary_table())


@dataclass
class AblationRunner:
    """
    Run systematic ablation studies across algorithms, benchmarks, and dimensions.

    Example:
        runner = AblationRunner(
            algorithms=["PSO", "HybridPSOFirefly"],
            benchmarks=["Rastrigin", "Ackley"],
            dimensions=[10, 30],
            runs_per_config=30,
        )
        results = runner.run()
        print(results.summary_table())
    """

    algorithms: list[str] = field(default_factory=lambda: ["PSO", "HybridPSOFirefly"])
    benchmarks: list[str] = field(default_factory=lambda: ["Rastrigin", "Ackley"])
    dimensions: list[int] = field(default_factory=lambda: [30])
    runs_per_config: int = 30
    n_particles: int = 50
    max_iterations: int = 1000

    def run(self) -> AblationResults:
        """Execute all configurations and collect results."""
        results = AblationResults()

        total = len(self.algorithms) * len(self.benchmarks) * len(self.dimensions) * self.runs_per_config
        completed = 0

        for algo_name in self.algorithms:
            algo_class = ALGORITHM_MAP.get(algo_name)
            if not algo_class:
                continue

            for bench_name in self.benchmarks:
                bench_class = BENCHMARKS.get(bench_name)
                if not bench_class:
                    continue

                bench_fn = bench_class()

                for dims in self.dimensions:
                    for run in range(self.runs_per_config):
                        optimizer = algo_class(
                            n_particles=self.n_particles,
                            dimensions=dims,
                            bounds=bench_fn.bounds,
                            max_iterations=self.max_iterations,
                            seed=run,
                        )

                        result = optimizer.optimize(bench_fn)
                        results.records.append({
                            "algorithm": algo_name,
                            "benchmark": bench_name,
                            "dimensions": dims,
                            "run": run,
                            "best_fitness": result.best_fitness,
                            "evaluations": result.evaluations,
                        })
                        completed += 1

        return results
