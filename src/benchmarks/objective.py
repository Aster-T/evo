"""连续目标封装：统一 FE 计数、best-so-far 收敛曲线与搜索边界。

算法只通过 `evaluate(X)` 访问目标函数；FE 记账、最优记录、预算判定都集中在此，
保证三算法对比公平（相同 FE 预算）、指标口径一致。最小化语义（越小越好）。
"""
from __future__ import annotations

import numpy as np

from .functions import FunctionSpec


def downsample_curve(curve: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """将逐 FE 的曲线下采样到 n_points，便于存储与绘图。"""
    L = len(curve)
    if L <= n_points:
        return curve.copy()
    idx = np.linspace(0, L - 1, n_points).astype(int)
    return curve[idx]


class ContinuousObjective:
    """把 FunctionSpec 包装成带预算/记账的可评估对象。"""

    def __init__(self, spec: FunctionSpec, dim: int, max_evals: int):
        self.spec = spec
        self.dim = dim
        self.lb = spec.lb
        self.ub = spec.ub
        self.max_evals = int(max_evals)

        self.fe = 0
        self.best_f = np.inf            # 最小化：越小越好
        self.best_x: np.ndarray | None = None
        self._curve = np.empty(self.max_evals, dtype=float)

    @property
    def done(self) -> bool:
        return self.fe >= self.max_evals

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """评估一批解。X 为 (n, d) 或 (d,)。返回 (n,) 或标量目标值（越小越好）。"""
        X = np.asarray(X, dtype=float)
        scalar = X.ndim == 1
        X2d = X[None, :] if scalar else X
        f = self.spec.func(X2d)

        for i in range(f.size):
            if self.fe >= self.max_evals:
                break
            if f[i] < self.best_f:
                self.best_f = float(f[i])
                self.best_x = X2d[i].copy()
            self._curve[self.fe] = self.best_f
            self.fe += 1

        return float(f[0]) if scalar else f

    def convergence_curve(self, n_points: int = 1000) -> np.ndarray:
        return downsample_curve(self._curve[:self.fe], n_points)
