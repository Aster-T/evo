"""多种群粒子群优化（mQSO 简化版），面向动态优化。

设计要点（Blackwell & Branke 的 mQSO 思路）：
  - M 个子群，每群 s 个粒子；每群一半为"量子"粒子，每步在该群 gbest 周围
    半径 r_cloud 的球内随机重置，持续提供多样性以应对环境变化；
  - 另一半为"中性"粒子，按带收缩因子的标准 PSO 速度公式更新；
  - 排斥（exclusion）：当两个子群的 gbest 距离小于 r_excl 时，重启较差的子群，
    避免多个子群收敛到同一峰；
  - 变化响应：环境变化后重评所有粒子（pbest/gbest 记忆失效）。

每步对全部 M*s 个粒子做一次评估，故 pop_size = n_swarms * swarm_size。
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class PSO(Optimizer):
    name = "PSO"

    def __init__(self, dim, lb, ub, rng, n_swarms=5, swarm_size=10,
                 chi=0.7298, c1=2.05, c2=2.05, r_cloud=1.0, quantum_fraction=0.5, **kw):
        super().__init__(dim, lb, ub, rng, **kw)
        self.M = n_swarms
        self.s = swarm_size
        self.pop_size = n_swarms * swarm_size
        self.chi = chi
        self.c1 = c1
        self.c2 = c2
        self.r_cloud = r_cloud
        # 量子粒子掩码：每群前 q 个为量子粒子
        q = max(1, int(quantum_fraction * swarm_size))
        self.qmask = np.zeros((self.M, self.s), dtype=bool)
        self.qmask[:, :q] = True
        # 排斥半径（mQSO 标准公式）
        self.r_excl = (ub - lb) / (2.0 * self.M ** (1.0 / dim))

    # ------------------------------------------------------------------ #
    def _eval_grid(self, X: np.ndarray) -> np.ndarray:
        f = self._evaluate(X.reshape(-1, self.dim))
        return f.reshape(self.M, self.s)

    def _recompute_gbest(self) -> None:
        bi = np.argmax(self.pbest_f, axis=1)
        rows = np.arange(self.M)
        self.gbest = self.pbest[rows, bi].copy()
        self.gbest_f = self.pbest_f[rows, bi].copy()

    def _uniform_in_ball(self, shape_ms: tuple[int, int]) -> np.ndarray:
        M, s = shape_ms
        dirs = self.rng.standard_normal((M, s, self.dim))
        dirs /= (np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-12)
        radii = self.r_cloud * self.rng.random((M, s, 1)) ** (1.0 / self.dim)
        return dirs * radii

    def _init_population(self) -> None:
        self.X = self.rng.uniform(self.lb, self.ub, size=(self.M, self.s, self.dim))
        span = self.ub - self.lb
        self.V = self.rng.uniform(-0.1 * span, 0.1 * span, size=(self.M, self.s, self.dim))
        f = self._eval_grid(self.X)
        self.pbest = self.X.copy()
        self.pbest_f = f.copy()
        self._recompute_gbest()
        self._update_best(self.X.reshape(-1, self.dim), f.ravel())

    def _step(self) -> None:
        gb = self.gbest[:, None, :]  # (M,1,d)
        r1 = self.rng.random((self.M, self.s, self.dim))
        r2 = self.rng.random((self.M, self.s, self.dim))
        V_new = self.chi * (self.V + self.c1 * r1 * (self.pbest - self.X)
                            + self.c2 * r2 * (gb - self.X))
        X_neutral = self.X + V_new
        X_quantum = gb + self._uniform_in_ball((self.M, self.s))

        qmask3 = self.qmask[:, :, None]
        self.X = np.where(qmask3, X_quantum, X_neutral)
        self.V = np.where(qmask3, 0.0, V_new)
        self.X = np.clip(self.X, self.lb, self.ub)

        f = self._eval_grid(self.X)
        better = f > self.pbest_f
        self.pbest = np.where(better[:, :, None], self.X, self.pbest)
        self.pbest_f = np.where(better, f, self.pbest_f)
        self._recompute_gbest()
        self._update_best(self.X.reshape(-1, self.dim), f.ravel())
        self._exclusion()

    def _exclusion(self) -> None:
        for i in range(self.M):
            for j in range(i + 1, self.M):
                if np.linalg.norm(self.gbest[i] - self.gbest[j]) < self.r_excl:
                    worse = i if self.gbest_f[i] < self.gbest_f[j] else j
                    self._reinit_swarm(worse)

    def _reinit_swarm(self, k: int) -> None:
        span = self.ub - self.lb
        self.X[k] = self.rng.uniform(self.lb, self.ub, size=(self.s, self.dim))
        self.V[k] = self.rng.uniform(-0.1 * span, 0.1 * span, size=(self.s, self.dim))
        f = self._evaluate(self.X[k])
        self.pbest[k] = self.X[k].copy()
        self.pbest_f[k] = f
        bi = int(np.argmax(f))
        self.gbest[k] = self.X[k, bi].copy()
        self.gbest_f[k] = float(f[bi])

    def _react(self) -> None:
        # 环境变化：重评全部粒子，重置记忆
        f = self._eval_grid(self.X)
        self.pbest = self.X.copy()
        self.pbest_f = f.copy()
        self._recompute_gbest()
        self.best_f = -np.inf
        self.best_x = None
        self._update_best(self.X.reshape(-1, self.dim), f.ravel())
