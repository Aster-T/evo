"""排列编码遗传算法（TSP）。

算子：二元锦标赛选择 + 顺序交叉 OX + 反转变异(2-opt 式片段翻转) + 精英保留。
目标：最小化巡回长度。
"""
from __future__ import annotations

import numpy as np

from ..common import TSPOptimizer


def order_crossover(p1: np.ndarray, p2: np.ndarray, rng) -> np.ndarray:
    """OX：保留 p1 的一段，其余按 p2 顺序填充。"""
    n = len(p1)
    a, b = np.sort(rng.integers(0, n, 2))
    child = -np.ones(n, dtype=int)
    child[a:b] = p1[a:b]
    chosen = set(p1[a:b].tolist())
    fill = [c for c in p2 if c not in chosen]
    pos = list(range(b, n)) + list(range(0, a))
    for k, i in enumerate(pos):
        child[i] = fill[k]
    return child


class GA(TSPOptimizer):
    name = "GA"

    def __init__(self, problem, rng, pop_size=100, crossover_prob=0.9,
                 mutation_prob=0.2, tournament=3, **kw):
        super().__init__(problem, rng, **kw)
        self.pop_size = pop_size
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.t = tournament

    def _init(self) -> None:
        self.pop = np.array([self.rng.permutation(self.n) for _ in range(self.pop_size)])
        self.fit = np.array([self.p.evaluate(t) for t in self.pop])

    def _select(self) -> np.ndarray:
        idx = self.rng.integers(0, self.pop_size, (self.pop_size, self.t))
        win = idx[np.arange(self.pop_size), np.argmin(self.fit[idx], axis=1)]
        return win

    def _inversion(self, tour: np.ndarray) -> np.ndarray:
        a, b = np.sort(self.rng.integers(0, self.n, 2))
        tour[a:b + 1] = tour[a:b + 1][::-1]
        return tour

    def _step(self) -> None:
        sel = self._select()
        offspring = []
        for k in range(self.pop_size):
            p1 = self.pop[sel[k]]
            if self.rng.random() < self.pc:
                p2 = self.pop[sel[self.rng.integers(self.pop_size)]]
                child = order_crossover(p1, p2, self.rng)
            else:
                child = p1.copy()
            if self.rng.random() < self.pm:
                child = self._inversion(child)
            offspring.append(child)
        offspring = np.array(offspring)
        fit = np.array([self.p.evaluate(t) for t in offspring])
        # 精英：保留全局最优巡回
        if self.p.best_tour is not None:
            worst = int(np.argmax(fit))
            offspring[worst] = self.p.best_tour
            fit[worst] = self.p.best_len
        self.pop, self.fit = offspring, fit
