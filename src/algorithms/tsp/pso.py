"""离散粒子群优化（TSP，交换序列 PSO，Clerc 2004）。

把排列视作粒子位置，**速度=交换算子序列**。位置向 pbest / gbest 学习，即沿
"把当前排列变成 pbest/gbest 所需的交换序列"按概率施加交换；惯性=保留部分历史交换。
连续 PSO 的速度概念在排列空间被离散化，目标：最小化巡回长度。
"""
from __future__ import annotations

import numpy as np

from ..common import TSPOptimizer


def swaps_to(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """把排列 a 变成 b 所需的交换序列（选择排序式）。"""
    a = a.copy()
    pos = {int(c): i for i, c in enumerate(a)}
    seq = []
    for i in range(len(a)):
        if a[i] != b[i]:
            j = pos[int(b[i])]
            pos[int(a[i])], pos[int(a[j])] = j, i
            a[i], a[j] = a[j], a[i]
            seq.append((i, j))
    return seq


def apply_swaps(perm: np.ndarray, seq, prob: float, rng) -> np.ndarray:
    perm = perm.copy()
    for i, j in seq:
        if rng.random() < prob:
            perm[i], perm[j] = perm[j], perm[i]
    return perm


class PSO(TSPOptimizer):
    name = "PSO"

    def __init__(self, problem, rng, swarm_size=50, w=0.3, c1=0.7, c2=0.9, **kw):
        super().__init__(problem, rng, **kw)
        self.s = swarm_size
        self.w, self.c1, self.c2 = w, c1, c2

    def _init(self) -> None:
        self.X = [self.rng.permutation(self.n) for _ in range(self.s)]
        self.fit = np.array([self.p.evaluate(t) for t in self.X])
        self.pbest = [x.copy() for x in self.X]
        self.pbest_f = self.fit.copy()
        # 速度（交换序列），初始化为若干随机交换
        self.V = [[] for _ in range(self.s)]
        g = int(np.argmin(self.fit))
        self.gbest = self.X[g].copy()
        self.gbest_f = float(self.fit[g])

    def _step(self) -> None:
        for k in range(self.s):
            x = self.X[k]
            # 惯性：按 w 保留历史交换
            v = [sw for sw in self.V[k] if self.rng.random() < self.w]
            x = apply_swaps(x, v, 1.0, self.rng)
            # 认知：向 pbest 学习
            cog = swaps_to(x, self.pbest[k])
            x = apply_swaps(x, cog, self.c1, self.rng)
            # 社会：向 gbest 学习
            soc = swaps_to(x, self.gbest)
            x = apply_swaps(x, soc, self.c2, self.rng)
            self.V[k] = v + cog + soc
            self.X[k] = x
            L = self.p.evaluate(x)
            self.fit[k] = L
            if L < self.pbest_f[k]:
                self.pbest[k] = x.copy()
                self.pbest_f[k] = L
                if L < self.gbest_f:
                    self.gbest = x.copy()
                    self.gbest_f = L
