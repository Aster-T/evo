"""算法基类：统一「按 FE 预算循环」的主循环。

连续与 TSP 两类问题各有一个轻量基类，子类只需实现 `_init` 与 `_step`。
约定：每个 `_step()` 评估一批解（一代）；目标/问题对象负责 FE 计数与 best 记录，
当预算耗尽（`obj.done`）时主循环停止。所有算法均为**最小化**。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ContinuousOptimizer(ABC):
    name = "base"

    def __init__(self, obj, rng: np.random.Generator, **params):
        self.obj = obj
        self.rng = rng
        self.dim = obj.dim
        self.lb = obj.lb
        self.ub = obj.ub
        self.params = params

    def run(self):
        self._init()
        while not self.obj.done:
            self._step()
        return self.obj

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lb, self.ub)

    def _rand_pop(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, size=(n, self.dim))

    @property
    def progress(self) -> float:
        """已用 FE 比例 ∈ [0,1]，用于线性调度参数（如 PSO 惯性权重）。"""
        return min(1.0, self.obj.fe / self.obj.max_evals)

    @abstractmethod
    def _init(self) -> None: ...

    @abstractmethod
    def _step(self) -> None: ...


class TSPOptimizer(ABC):
    name = "base"

    def __init__(self, problem, rng: np.random.Generator, **params):
        self.p = problem
        self.rng = rng
        self.n = problem.n
        self.D = problem.D
        self.params = params

    def run(self):
        self._init()
        while not self.p.done:
            self._step()
        return self.p

    @abstractmethod
    def _init(self) -> None: ...

    @abstractmethod
    def _step(self) -> None: ...
