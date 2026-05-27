"""优化器基类：统一主循环 + 变化响应钩子。

关键约定：**每个 _step() 恰好进行一次种群评估**（一批 pop_size 次函数评估），
因此整批评估落在同一环境内，环境变化总在步与步之间被检测到，
使得变化响应（重评 + 重启）逻辑干净，且 Offline Error 记账一致。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    name = "base"

    def __init__(self, dim: int, lb: float, ub: float, rng: np.random.Generator, **params):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.rng = rng
        self.params = params
        self.benchmark = None
        self.best_x: np.ndarray | None = None
        self.best_f: float = -np.inf

    # —— 工具 —— #
    def _rand_pop(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, size=(n, self.dim))

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """裁剪到边界并评估（计入 FE）。返回适应度（越大越好）。"""
        Xc = np.clip(X, self.lb, self.ub)
        f = self.benchmark.evaluate(Xc)
        return np.atleast_1d(f)

    def _update_best(self, X: np.ndarray, f: np.ndarray) -> None:
        i = int(np.argmax(f))
        if f[i] > self.best_f:
            self.best_f = float(f[i])
            self.best_x = X[i].copy()

    # —— 主循环 —— #
    def run(self, benchmark) -> object:
        self.benchmark = benchmark
        self._init_population()
        while not benchmark.done:
            self._step()
            if benchmark.changed:
                self._react()
                benchmark.acknowledge_change()
        return benchmark

    # —— 子类实现 —— #
    @abstractmethod
    def _init_population(self) -> None: ...

    @abstractmethod
    def _step(self) -> None: ...

    @abstractmethod
    def _react(self) -> None:
        """环境变化响应：重评陈旧适应度记忆 + 注入多样性。"""
        ...
