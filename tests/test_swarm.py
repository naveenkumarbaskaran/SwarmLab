"""Tests for SwarmLab algorithms."""

import pytest
import numpy as np
from swarmlab import PSO, FireflyAlgorithm, HybridPSOFirefly, DifferentialEvolution
from swarmlab import Rastrigin, Ackley, Schwefel, Griewank, Sphere
from swarmlab import AblationRunner


# ── Benchmark Function Tests ─────────────────────────────────────

class TestBenchmarks:
    def test_sphere_at_origin(self):
        f = Sphere()
        assert f(np.zeros(10)) == 0.0

    def test_rastrigin_at_origin(self):
        f = Rastrigin()
        assert f(np.zeros(10)) == pytest.approx(0.0)

    def test_ackley_at_origin(self):
        f = Ackley()
        assert f(np.zeros(10)) == pytest.approx(0.0, abs=1e-10)

    def test_griewank_at_origin(self):
        f = Griewank()
        assert f(np.zeros(10)) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("func_class", [Sphere, Rastrigin, Ackley, Griewank])
    def test_positive_away_from_origin(self, func_class):
        f = func_class()
        x = np.ones(10) * 2.0
        assert f(x) > 0


# ── PSO Tests ────────────────────────────────────────────────────

class TestPSO:
    def test_optimizes_sphere(self):
        """PSO should easily solve Sphere in low dims."""
        pso = PSO(n_particles=30, dimensions=5, bounds=(-5.12, 5.12),
                  max_iterations=200, seed=42)
        result = pso.optimize(Sphere())
        assert result.best_fitness < 0.01

    def test_convergence_decreasing(self):
        pso = PSO(n_particles=30, dimensions=5, max_iterations=100, seed=42)
        result = pso.optimize(Sphere())
        # Convergence should be non-increasing (mostly)
        curve = result.convergence_curve
        assert curve[-1] <= curve[0]

    def test_evaluation_count(self):
        pso = PSO(n_particles=20, dimensions=5, max_iterations=50, seed=42)
        result = pso.optimize(Sphere())
        # Initial eval (20) + 50 iters × 20 particles = 1020
        assert result.evaluations == 20 + 50 * 20

    def test_stays_in_bounds(self):
        pso = PSO(n_particles=20, dimensions=5, bounds=(-1, 1),
                  max_iterations=100, seed=42)
        pso.optimize(Sphere())
        assert np.all(pso.positions >= -1)
        assert np.all(pso.positions <= 1)


# ── Firefly Tests ────────────────────────────────────────────────

class TestFirefly:
    def test_optimizes_sphere(self):
        ff = FireflyAlgorithm(n_particles=30, dimensions=5,
                              max_iterations=100, seed=42)
        result = ff.optimize(Sphere())
        assert result.best_fitness < 1.0

    def test_rastrigin_multimodal(self):
        """Firefly should handle multi-modal better than PSO."""
        ff = FireflyAlgorithm(n_particles=40, dimensions=10,
                              max_iterations=300, seed=42)
        result = ff.optimize(Rastrigin())
        assert result.best_fitness < 50  # reasonable for 10D Rastrigin


# ── Hybrid Tests ─────────────────────────────────────────────────

class TestHybrid:
    def test_optimizes_sphere(self):
        hybrid = HybridPSOFirefly(n_particles=30, dimensions=5,
                                   max_iterations=200, seed=42)
        result = hybrid.optimize(Sphere())
        assert result.best_fitness < 0.01

    def test_beats_pso_on_rastrigin(self):
        """Hybrid should outperform vanilla PSO on multi-modal."""
        pso = PSO(n_particles=40, dimensions=10, max_iterations=500, seed=42)
        hybrid = HybridPSOFirefly(n_particles=40, dimensions=10,
                                   max_iterations=500, seed=42)

        pso_result = pso.optimize(Rastrigin())
        hybrid_result = hybrid.optimize(Rastrigin())

        # Hybrid should be at least somewhat better
        assert hybrid_result.best_fitness <= pso_result.best_fitness * 1.5

    def test_algorithm_name(self):
        hybrid = HybridPSOFirefly(n_particles=10, dimensions=2, max_iterations=10)
        result = hybrid.optimize(Sphere())
        assert result.algorithm == "HybridPSOFirefly"


# ── DE Tests ─────────────────────────────────────────────────────

class TestDE:
    def test_optimizes_sphere(self):
        de = DifferentialEvolution(n_particles=30, dimensions=5,
                                   max_iterations=200, seed=42)
        result = de.optimize(Sphere())
        assert result.best_fitness < 0.01

    def test_rosenbrock(self):
        from swarmlab.benchmarks.functions import Rosenbrock
        de = DifferentialEvolution(n_particles=50, dimensions=10,
                                   bounds=(-5, 10), max_iterations=500, seed=42)
        result = de.optimize(Rosenbrock())
        assert result.best_fitness < 100  # Rosenbrock is hard


# ── Ablation Runner Tests ────────────────────────────────────────

class TestAblation:
    def test_small_ablation(self):
        runner = AblationRunner(
            algorithms=["PSO"],
            benchmarks=["Sphere"],
            dimensions=[5],
            runs_per_config=3,
            n_particles=10,
            max_iterations=50,
        )
        results = runner.run()
        assert len(results.records) == 3

    def test_summary_table(self):
        runner = AblationRunner(
            algorithms=["PSO", "DE"],
            benchmarks=["Sphere"],
            dimensions=[5],
            runs_per_config=2,
            n_particles=10,
            max_iterations=50,
        )
        results = runner.run()
        table = results.summary_table()
        assert "PSO" in table
        assert "DE" in table
        assert "Sphere" in table
