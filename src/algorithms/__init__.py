"""三种进化算法（均为最大化器，针对动态优化做了变化响应）。

GMPB 中 f(x) 为待最大化的适应度（峰为极大值，全局最优 = max 峰高）。
所有算法只通过 Benchmark.evaluate 访问目标函数，并在 benchmark.changed 时执行
变化响应（重评历史 + 部分重启），以适配动态优化问题（DOP）。
"""
from .base import Optimizer
from .ga import GA
from .pso import PSO
from .de import DE

ALGORITHMS = {"GA": GA, "PSO": PSO, "DE": DE}

__all__ = ["Optimizer", "GA", "PSO", "DE", "ALGORITHMS"]
