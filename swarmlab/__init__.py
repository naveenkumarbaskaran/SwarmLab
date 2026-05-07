"""SwarmLab — Swarm Intelligence Simulation Framework."""

from swarmlab.algorithms.pso import PSO
from swarmlab.algorithms.firefly import FireflyAlgorithm
from swarmlab.algorithms.hybrid import HybridPSOFirefly
from swarmlab.algorithms.de import DifferentialEvolution
from swarmlab.benchmarks.functions import Rastrigin, Ackley, Schwefel, Griewank, Sphere
from swarmlab.analysis.ablation import AblationRunner

__version__ = "2.0.0"
__all__ = [
    "PSO", "FireflyAlgorithm", "HybridPSOFirefly", "DifferentialEvolution",
    "Rastrigin", "Ackley", "Schwefel", "Griewank", "Sphere",
    "AblationRunner",
]
