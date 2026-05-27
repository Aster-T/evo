"""聚类（簇中心优化）问题（实数编码，复用 GA/PSO/DE）。

决策变量为 K 个簇中心展平：``c ∈ R^{K·D}``，reshape 为 (K, D)。
目标（最小化类内平方误差 SSE）：
    SSE(C) = Σ_i min_k ‖x_i − c_k‖²
对标准化后的特征优化；边界取数据全局 [min, max]（标量，统一各维）。
事后用真实质量评分计算 ARI / NMI 作为外部聚类评价。
"""
from __future__ import annotations

import numpy as np

from .static import StaticObjective


def _assign(X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    """对单组中心 C (K, D)：返回样本所属簇标签与 SSE。"""
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (n, K)
    labels = d2.argmin(axis=1)
    sse = float(d2.min(axis=1).sum())
    return labels, sse


class ClusteringProblem:
    """K-means 式簇中心优化，向优化器暴露 StaticObjective。"""

    def __init__(self, X: np.ndarray, K: int, *, labels_true: np.ndarray | None = None,
                 max_evals: int = 30000):
        self.X = np.asarray(X, dtype=float)
        self.n, self.D = self.X.shape
        self.K = K
        self.labels_true = labels_true
        self.dim = K * self.D
        # 边界取 1–99 百分位（标准化特征），排除离群点以收缩高维搜索空间
        self.lb = float(np.percentile(self.X, 1))
        self.ub = float(np.percentile(self.X, 99))
        self.max_evals = max_evals

    def _sse_batch(self, g: np.ndarray) -> np.ndarray:
        """g: (p, K*D) → 每个候选中心组的 SSE (p,)。逐候选计算以控制内存。"""
        g = np.atleast_2d(g)
        p = g.shape[0]
        out = np.empty(p)
        for i in range(p):
            C = g[i].reshape(self.K, self.D)
            d2 = ((self.X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            out[i] = d2.min(axis=1).sum()
        return out

    def make_objective(self) -> StaticObjective:
        return StaticObjective(self._sse_batch, dim=self.dim, lb=self.lb,
                               ub=self.ub, max_evals=self.max_evals, maximize=False)

    def metrics_of(self, g_best: np.ndarray) -> dict:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        C = g_best.reshape(self.K, self.D)
        labels, sse = _assign(self.X, C)
        out = {"sse": sse}
        if self.labels_true is not None:
            out["ari"] = float(adjusted_rand_score(self.labels_true, labels))
            out["nmi"] = float(normalized_mutual_info_score(self.labels_true, labels))
        return out
