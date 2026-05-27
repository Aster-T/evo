"""投资组合优化问题（实数编码，复用 GA/PSO/DE）。

决策变量为 genotype ``g ∈ [lb, ub]^N``，经 **softmax 解码**为权重
``w = softmax(g)``，自动满足 ``Σ w_i = 1`` 与 ``w_i ≥ 0``（全额投资、禁卖空）。
可选**基数约束 K**：解码后仅保留权重最大的 K 个资产，其余置零再归一化
（``‖w‖_0 ≤ K``，NP-hard，制造三算法的性能差异）。

目标（默认最大化夏普比率）：
    Sharpe(w) = (wᵀμ − r_f) / sqrt(wᵀΣw)
内部以 ``neg_sharpe = −Sharpe`` 作为**最小化**目标（与 stats.py「越小越好」一致）。
也支持最小方差目标 ``wᵀΣw``。
"""
from __future__ import annotations

import numpy as np

from .static import StaticObjective

_EPS = 1e-12


def softmax_decode(g: np.ndarray) -> np.ndarray:
    """行 softmax：(n, N) genotype → (n, N) 权重，每行非负且和为 1。"""
    g = np.atleast_2d(g)
    z = g - g.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + _EPS)


def apply_cardinality(W: np.ndarray, K: int) -> np.ndarray:
    """每行仅保留权重最大的 K 个，其余置零后重新归一化。"""
    n, N = W.shape
    if K is None or K >= N:
        return W
    out = np.zeros_like(W)
    # 每行 top-K 索引
    topk = np.argpartition(-W, K - 1, axis=1)[:, :K]
    rows = np.arange(n)[:, None]
    out[rows, topk] = W[rows, topk]
    out /= (out.sum(axis=1, keepdims=True) + _EPS)
    return out


class PortfolioProblem:
    """投资组合优化问题，向优化器暴露 StaticObjective。"""

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, *, K: int | None = None,
                 r_f: float = 0.0, objective: str = "sharpe",
                 lb: float = -5.0, ub: float = 5.0, max_evals: int = 30000):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.N = self.mu.size
        self.K = K
        self.r_f = r_f
        self.objective = objective
        self.lb, self.ub, self.max_evals = lb, ub, max_evals

    # —— 解码与目标 —— #
    def decode(self, g: np.ndarray) -> np.ndarray:
        """genotype → 投资权重（含基数约束）。"""
        return apply_cardinality(softmax_decode(g), self.K)

    def _neg_sharpe(self, W: np.ndarray) -> np.ndarray:
        ret = W @ self.mu
        var = np.einsum("ij,jk,ik->i", W, self.Sigma, W)
        sharpe = (ret - self.r_f) / np.sqrt(np.maximum(var, _EPS))
        return -sharpe

    def _variance(self, W: np.ndarray) -> np.ndarray:
        return np.einsum("ij,jk,ik->i", W, self.Sigma, W)

    def _objective_batch(self, g: np.ndarray) -> np.ndarray:
        W = self.decode(g)
        return self._variance(W) if self.objective == "min_variance" else self._neg_sharpe(W)

    def make_objective(self) -> StaticObjective:
        """构造供优化器调用的静态目标（最小化 neg_sharpe / variance）。"""
        return StaticObjective(self._objective_batch, dim=self.N, lb=self.lb,
                               ub=self.ub, max_evals=self.max_evals, maximize=False)

    # —— 由最优 genotype 计算可读指标 —— #
    def metrics_of(self, g_best: np.ndarray) -> dict:
        w = self.decode(g_best.reshape(1, -1))[0]
        ret = float(w @ self.mu)
        var = float(w @ self.Sigma @ w)
        sharpe = (ret - self.r_f) / np.sqrt(max(var, _EPS))
        return {"sharpe": sharpe, "neg_sharpe": -sharpe, "ret": ret,
                "variance": var, "cardinality": int((w > 1e-6).sum()), "weights": w}
