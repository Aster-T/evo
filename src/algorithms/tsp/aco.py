"""蚁群算法（TSP，Ant System，Dorigo 1996）。

信息素矩阵 τ 与启发式 η=1/d_ij 共同引导蚂蚁概率式地构造巡回：
    p(i→j) ∝ τ_ij^α · η_ij^β   （仅在未访问城市间）
每代所有蚂蚁构造完毕后：信息素蒸发 τ←(1-ρ)τ，再按各蚂蚁巡回质量沉积 Δτ=Q/L_k。
这是 ACO 的本职问题（组合优化），目标：最小化巡回长度。
"""
from __future__ import annotations

import numpy as np

from ..common import TSPOptimizer


class ACO(TSPOptimizer):
    name = "ACO"

    def __init__(self, problem, rng, n_ants=None, alpha=1.0, beta=2.0,
                 rho=0.5, Q=1.0, **kw):
        super().__init__(problem, rng, **kw)
        self.m = n_ants or self.n
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        with np.errstate(divide="ignore"):
            self.eta = 1.0 / self.D
        np.fill_diagonal(self.eta, 0.0)

    def _init(self) -> None:
        # 信息素初值：贪心最近邻巡回长度的倒数尺度（经典启发式）
        nn = self._nearest_neighbor_length()
        self.tau = np.full((self.n, self.n), self.m / max(nn, 1e-9))
        np.fill_diagonal(self.tau, 0.0)

    def _nearest_neighbor_length(self) -> float:
        start = 0
        visited = np.zeros(self.n, dtype=bool)
        visited[start] = True
        cur, total = start, 0.0
        for _ in range(self.n - 1):
            d = self.D[cur].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            total += self.D[cur, nxt]
            visited[nxt] = True
            cur = nxt
        return total + self.D[cur, start]

    def _construct(self) -> np.ndarray:
        """所有蚂蚁并行构造巡回，返回 (m, n) 排列。"""
        tau_b = self.tau ** self.alpha
        eta_b = self.eta ** self.beta
        attract = tau_b * eta_b                       # (n, n)
        tours = np.empty((self.m, self.n), dtype=int)
        start = self.rng.integers(0, self.n, self.m)
        tours[:, 0] = start
        visited = np.zeros((self.m, self.n), dtype=bool)
        visited[np.arange(self.m), start] = True
        cur = start.copy()
        for step in range(1, self.n):
            w = attract[cur]                          # (m, n)
            w[visited] = 0.0
            s = w.sum(axis=1, keepdims=True)
            # 退化保护：若全 0（数值下溢）则在未访问中均匀选
            zero = s[:, 0] <= 0
            if zero.any():
                w[zero] = (~visited[zero]).astype(float)
                s = w.sum(axis=1, keepdims=True)
            prob = w / s
            r = self.rng.random((self.m, 1))
            nxt = (prob.cumsum(axis=1) >= r).argmax(axis=1)
            tours[:, step] = nxt
            visited[np.arange(self.m), nxt] = True
            cur = nxt
        return tours

    def _step(self) -> None:
        tours = self._construct()
        lengths = np.array([self.p.evaluate(t) for t in tours])
        self.tau *= (1.0 - self.rho)                  # 蒸发
        for t, L in zip(tours, lengths):              # 沉积
            nxt = np.roll(t, -1)
            self.tau[t, nxt] += self.Q / L
            self.tau[nxt, t] += self.Q / L            # 对称
