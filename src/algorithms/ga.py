"""实数编码遗传算法（GA），针对 DOP 加入变化检测后的部分重启。

算子：二元锦标赛选择 + SBX 模拟二进制交叉 + 多项式变异 + 精英保留。
变化响应：重评当前种群（旧适应度已失效）+ 随机重启一部分个体维持多样性。
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class GA(Optimizer):
    name = "GA"

    def __init__(self, dim, lb, ub, rng, pop_size=50, crossover_prob=0.9,
                 eta_c=15.0, eta_m=20.0, mutation_prob=None, restart_fraction=0.5, **kw):
        super().__init__(dim, lb, ub, rng, **kw)
        self.pop_size = pop_size
        self.pc = crossover_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.pm = mutation_prob if mutation_prob is not None else 1.0 / dim
        self.restart_fraction = restart_fraction

    # ------------------------------------------------------------------ #
    def _init_population(self) -> None:
        self.X = self._rand_pop(self.pop_size)
        self.fx = self._evaluate(self.X)
        self._update_best(self.X, self.fx)

    def _tournament(self) -> np.ndarray:
        a = self.rng.integers(0, self.pop_size, self.pop_size)
        b = self.rng.integers(0, self.pop_size, self.pop_size)
        win = np.where(self.fx[a] >= self.fx[b], a, b)
        return self.X[win].copy()

    def _sbx(self, parents: np.ndarray) -> np.ndarray:
        p = parents.copy()
        self.rng.shuffle(p)
        n = self.pop_size - (self.pop_size % 2)
        children = p.copy()
        for i in range(0, n, 2):
            p1, p2 = p[i], p[i + 1]
            if self.rng.random() < self.pc:
                u = self.rng.random(self.dim)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta_c + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
                children[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                children[i + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return children

    def _poly_mutation(self, X: np.ndarray) -> np.ndarray:
        span = self.ub - self.lb
        mask = self.rng.random(X.shape) < self.pm
        u = self.rng.random(X.shape)
        delta = np.where(u < 0.5,
                         (2 * u) ** (1 / (self.eta_m + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
        out = X.copy()
        out[mask] = X[mask] + delta[mask] * span
        return out

    def _step(self) -> None:
        parents = self._tournament()
        offspring = self._sbx(parents)
        offspring = self._poly_mutation(offspring)
        offspring = np.clip(offspring, self.lb, self.ub)
        f_off = self._evaluate(offspring)

        # 精英保留：用全局最优个体替换子代中最差者
        if self.best_x is not None:
            worst = int(np.argmin(f_off))
            offspring[worst] = self.best_x
            f_off[worst] = self.best_f

        self.X, self.fx = offspring, f_off
        self._update_best(self.X, self.fx)

    def _react(self) -> None:
        # 旧适应度失效 → 重评；随机重启一部分个体
        k = int(self.restart_fraction * self.pop_size)
        if k > 0:
            idx = self.rng.choice(self.pop_size, k, replace=False)
            self.X[idx] = self._rand_pop(k)
        self.best_f = -np.inf  # 旧 best 失效
        self.best_x = None
        self.fx = self._evaluate(self.X)
        self._update_best(self.X, self.fx)
