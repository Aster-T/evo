"""实数编码遗传算法（连续测试函数）。

算子：二元锦标赛选择 + 模拟二进制交叉 SBX + 多项式变异 + 精英保留。
最小化目标 f（越小越好）。
"""
from __future__ import annotations

import numpy as np

from ..common import ContinuousOptimizer


class GA(ContinuousOptimizer):
    name = "GA"

    def __init__(self, obj, rng, pop_size=50, crossover_prob=0.9,
                 eta_c=15.0, eta_m=20.0, mutation_prob=None, **kw):
        super().__init__(obj, rng, **kw)
        self.pop_size = pop_size
        self.pc = crossover_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.pm = mutation_prob if mutation_prob is not None else 1.0 / self.dim

    def _init(self) -> None:
        self.X = self._rand_pop(self.pop_size)
        self.fx = self.obj.evaluate(self.X)

    def _tournament(self) -> np.ndarray:
        i = self.rng.integers(0, self.pop_size, self.pop_size)
        j = self.rng.integers(0, self.pop_size, self.pop_size)
        win = np.where(self.fx[i] <= self.fx[j], i, j)
        return self.X[win].copy()

    def _sbx(self, P: np.ndarray) -> np.ndarray:
        Q = P.copy()
        half = self.pop_size // 2
        a, b = Q[:half], Q[half:2 * half]
        u = self.rng.random(a.shape)
        beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (self.eta_c + 1)),
                        (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1)))
        do = self.rng.random(a.shape[0])[:, None] < self.pc
        c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
        c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
        Q[:half] = np.where(do, c1, a)
        Q[half:2 * half] = np.where(do, c2, b)
        return Q

    def _poly_mutation(self, P: np.ndarray) -> np.ndarray:
        span = self.ub - self.lb
        mask = self.rng.random(P.shape) < self.pm
        u = self.rng.random(P.shape)
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (self.eta_m + 1)) - 1.0,
                         1.0 - (2 * (1 - u)) ** (1.0 / (self.eta_m + 1)))
        return np.where(mask, P + delta * span, P)

    def _step(self) -> None:
        P = self._tournament()
        Q = self._sbx(P)
        Q = self._poly_mutation(Q)
        Q = self._clip(Q)
        fq = self.obj.evaluate(Q)
        # 精英：用全局最优替换本代最差个体
        worst = int(np.argmax(fq))
        if self.obj.best_x is not None:
            Q[worst] = self.obj.best_x
            fq[worst] = self.obj.best_f
        self.X, self.fx = Q, fq
