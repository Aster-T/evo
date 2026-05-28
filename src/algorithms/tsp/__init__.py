"""TSP 上的三种算法（排列/离散编码）。"""
from .aco import ACO
from .ga import GA
from .pso import PSO

TSP_ALGORITHMS = {"GA": GA, "PSO": PSO, "ACO": ACO}

__all__ = ["GA", "PSO", "ACO", "TSP_ALGORITHMS"]
