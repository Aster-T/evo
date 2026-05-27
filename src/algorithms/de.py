"""差分进化（DE/rand/1/bin），针对 DOP 加入变化检测后的部分重启。

变异：v = x_r1 + F*(x_r2 - x_r3)；二项交叉；贪婪选择。
变化响应：重评当前种群 + 随机重启一部分个体。
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class DE(Optimizer):
    name = "DE"

    def __init__(self, dim, lb, ub, rng, pop_size=50, F=0.5, CR=0.9,
                 restart_fraction=0.3, **kw):
        super().__init__(dim, lb, ub, rng, **kw)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.restart_fraction = restart_fraction

    def _init_population(self) -> None:
        self.X = self._rand_pop(self.pop_size)
        self.fx = self._evaluate(self.X)
        self._update_best(self.X, self.fx)

    def _distinct_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.pop_size
        idx = np.arange(n)
        r = np.empty((3, n), dtype=int)
        for j in range(3):
            cand = self.rng.integers(0, n, n)
            # 保证与 i 及彼此不同
            for _ in range(8):
                clash = (cand == idx)
                for jj in range(j):
                    clash |= (cand == r[jj])
                if not clash.any():
                    break
                cand[clash] = self.rng.integers(0, n, clash.sum())
            r[j] = cand
        return r[0], r[1], r[2]

    def _step(self) -> None:
        r1, r2, r3 = self._distinct_indices()
        V = self.X[r1] + self.F * (self.X[r2] - self.X[r3])
        # 二项交叉
        cross = self.rng.random((self.pop_size, self.dim)) < self.CR
        jrand = self.rng.integers(0, self.dim, self.pop_size)
        cross[np.arange(self.pop_size), jrand] = True
        U = np.where(cross, V, self.X)
        U = np.clip(U, self.lb, self.ub)

        f_u = self._evaluate(U)
        improved = f_u >= self.fx
        self.X[improved] = U[improved]
        self.fx[improved] = f_u[improved]
        self._update_best(self.X, self.fx)

    def _react(self) -> None:
        k = int(self.restart_fraction * self.pop_size)
        if k > 0:
            idx = self.rng.choice(self.pop_size, k, replace=False)
            self.X[idx] = self._rand_pop(k)
        self.best_f = -np.inf
        self.best_x = None
        self.fx = self._evaluate(self.X)
        self._update_best(self.X, self.fx)
