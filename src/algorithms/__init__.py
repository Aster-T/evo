"""GA / ACO / PSO 三种算法，分连续（测试函数）与组合（TSP）两种形态。

- 连续：GA(实数) / PSO(标准) / ACO=ACOR(连续蚁群)  —— src.algorithms.continuous
- TSP ：GA(排列) / PSO(交换序列) / ACO(Ant System) —— src.algorithms.tsp
"""
from .continuous import CONTINUOUS_ALGORITHMS
from .tsp import TSP_ALGORITHMS

ALGORITHM_NAMES = ["GA", "ACO", "PSO"]

__all__ = ["CONTINUOUS_ALGORITHMS", "TSP_ALGORITHMS", "ALGORITHM_NAMES"]
