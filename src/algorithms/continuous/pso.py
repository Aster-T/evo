"""标准粒子群优化（连续测试函数）。

全局最优（gbest）拓扑；惯性权重 w 随 FE 进度线性 0.9→0.4；速度钳制 v_max=0.2·域宽。
最小化目标。
"""
from __future__ import annotations

import numpy as np

from ..common import ContinuousOptimizer


class PSO(ContinuousOptimizer):
    name = "PSO"

    def __init__(self, obj, rng, swarm_size=50, w_max=0.9, w_min=0.4,
                 c1=2.0, c2=2.0, v_frac=0.2, **kw):
        super().__init__(obj, rng, **kw)
        self.s = swarm_size
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.vmax = v_frac * (self.ub - self.lb)

    def _init(self) -> None:
        self.X = self._rand_pop(self.s)
        self.V = self.rng.uniform(-self.vmax, self.vmax, size=(self.s, self.dim))
        f = self.obj.evaluate(self.X)
        self.pbest = self.X.copy()
        self.pbest_f = f.copy()
        g = int(np.argmin(f))
        self.gbest = self.X[g].copy()
        self.gbest_f = float(f[g])

    def _step(self) -> None:
        w = self.w_max - (self.w_max - self.w_min) * self.progress
        r1 = self.rng.random((self.s, self.dim))
        r2 = self.rng.random((self.s, self.dim))
        self.V = (w * self.V
                  + self.c1 * r1 * (self.pbest - self.X)
                  + self.c2 * r2 * (self.gbest - self.X))
        self.V = np.clip(self.V, -self.vmax, self.vmax)
        self.X = self._clip(self.X + self.V)

        f = self.obj.evaluate(self.X)
        better = f < self.pbest_f
        self.pbest[better] = self.X[better]
        self.pbest_f[better] = f[better]
        g = int(np.argmin(self.pbest_f))
        if self.pbest_f[g] < self.gbest_f:
            self.gbest = self.pbest[g].copy()
            self.gbest_f = float(self.pbest_f[g])
