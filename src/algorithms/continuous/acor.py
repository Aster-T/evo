"""ACOR —— 连续域蚁群优化（Socha & Dorigo, 2008）。

把「信息素」表示为一个大小为 k 的**解档案**（按目标值升序排列）。每只蚂蚁逐维构造解：
按 rank 权重选一个引导解 l，以该解第 i 维为均值、以它到档案中其他解的平均距离×ξ 为标准差，
从高斯分布采样。新蚂蚁与档案合并后保留最好的 k 个。是 ACO 思想在连续空间的标准推广，
使蚁群算法能直接优化 Appendix C 的连续测试函数。最小化目标。
"""
from __future__ import annotations

import numpy as np

from ..common import ContinuousOptimizer


class ACOR(ContinuousOptimizer):
    name = "ACO"  # 报告中与 TSP 的经典蚁群统一记为 ACO（连续形态 = ACOR）

    def __init__(self, obj, rng, archive_size=50, n_ants=10, q=1e-4, xi=0.85, **kw):
        super().__init__(obj, rng, **kw)
        self.k = archive_size
        self.m = n_ants
        self.q = q
        self.xi = xi
        # rank 权重 w_l（l=1..k），概率 p_l ∝ w_l
        l = np.arange(1, self.k + 1)
        w = (1.0 / (q * self.k * np.sqrt(2 * np.pi))
             * np.exp(-((l - 1) ** 2) / (2.0 * (q * self.k) ** 2)))
        self.p = w / w.sum()

    def _init(self) -> None:
        A = self._rand_pop(self.k)
        fA = self.obj.evaluate(A)
        order = np.argsort(fA)
        self.A = A[order]
        self.fA = fA[order]

    def _sigma(self) -> np.ndarray:
        """档案每个解每一维的标准差 σ[l,i] = ξ/(k-1) Σ_e |A[e,i]-A[l,i]|。"""
        diff = np.abs(self.A[None, :, :] - self.A[:, None, :])  # (k,k,d)
        return self.xi / (self.k - 1) * diff.sum(axis=1)        # (k,d)

    def _step(self) -> None:
        sigma = self._sigma()
        cols = np.arange(self.dim)[None, :]
        # 每 (蚂蚁, 维) 独立按 p 选引导解
        idx = self.rng.choice(self.k, size=(self.m, self.dim), p=self.p)
        mu = self.A[idx, cols]
        sd = sigma[idx, cols]
        ants = self._clip(mu + sd * self.rng.standard_normal((self.m, self.dim)))
        fa = self.obj.evaluate(ants)

        # 合并档案并保留最优 k
        A = np.vstack([self.A, ants])
        fA = np.concatenate([self.fA, fa])
        order = np.argsort(fA)[:self.k]
        self.A, self.fA = A[order], fA[order]
