"""连续测试函数上的三种算法（实数编码）。"""
from .acor import ACOR
from .ga import GA
from .pso import PSO

CONTINUOUS_ALGORITHMS = {"GA": GA, "PSO": PSO, "ACO": ACOR}

__all__ = ["GA", "PSO", "ACOR", "CONTINUOUS_ALGORITHMS"]
