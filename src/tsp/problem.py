"""TSP 问题封装：距离矩阵 + 巡回长度评估 + FE 计数 + best-so-far 曲线。

巡回为城市访问排列 π（闭合回路）；长度 = Σ d(π_i, π_{i+1}) + d(π_n, π_1)。
算法只通过 `tour_length` / `evaluate` 访问目标，FE 记账集中于此，保证公平对比。
最小化语义（巡回越短越好）。
"""
from __future__ import annotations

import numpy as np

from .data import OPTIMA, distance_matrix, load_coords


def downsample_curve(curve: np.ndarray, n_points: int = 1000) -> np.ndarray:
    L = len(curve)
    if L <= n_points:
        return curve.copy()
    idx = np.linspace(0, L - 1, n_points).astype(int)
    return curve[idx]


class TSPProblem:
    """单个 TSPLIB 实例的可评估封装。"""

    def __init__(self, instance: str, max_evals: int):
        self.instance = instance
        self.coords = load_coords(instance)
        self.n = len(self.coords)
        self.D = distance_matrix(self.coords)
        self.optimum = OPTIMA.get(instance)
        self.max_evals = int(max_evals)

        self.fe = 0
        self.best_len = np.inf
        self.best_tour: np.ndarray | None = None
        self._curve = np.empty(self.max_evals, dtype=float)

    @property
    def done(self) -> bool:
        return self.fe >= self.max_evals

    def tour_length(self, tour: np.ndarray) -> float:
        """巡回长度（不计 FE，供算子内部使用，如局部搜索增量评估）。"""
        idx = np.asarray(tour, dtype=int)
        nxt = np.roll(idx, -1)
        return float(self.D[idx, nxt].sum())

    def evaluate(self, tour: np.ndarray) -> float:
        """评估一条巡回并计入 FE 预算、更新 best-so-far。"""
        L = self.tour_length(tour)
        if self.fe < self.max_evals:
            if L < self.best_len:
                self.best_len = L
                self.best_tour = np.asarray(tour, dtype=int).copy()
            self._curve[self.fe] = self.best_len
            self.fe += 1
        return L

    def gap(self, length: float | None = None) -> float:
        """相对已知最优的 gap%（length 缺省用 best_len）。"""
        if self.optimum is None:
            return float("nan")
        L = self.best_len if length is None else length
        return 100.0 * (L - self.optimum) / self.optimum

    def convergence_curve(self, n_points: int = 1000) -> np.ndarray:
        return downsample_curve(self._curve[:self.fe], n_points)
