"""基线方法：为进化算法提供参照（ground truth / 经典启发式）。

投资组合：
  * qp_max_sharpe —— scipy SLSQP 在「Σw=1, w≥0」凸可行域上最大化夏普比率。
    无基数约束时此问题接近凸，给出近似全局最优，可衡量 EA 的最优性 gap。
  * equal_weight  —— 等权组合 1/N（朴素基线）。
聚类：
  * kmeans_pp —— sklearn KMeans（k-means++ 初始化，多次重启取最优 SSE），强基线。
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------- 投资组合
def _neg_sharpe_w(w, mu, Sigma, r_f):
    ret = w @ mu
    var = w @ Sigma @ w
    return -(ret - r_f) / np.sqrt(max(var, _EPS))


def qp_max_sharpe(mu: np.ndarray, Sigma: np.ndarray, r_f: float = 0.0) -> dict:
    """SLSQP 求「全额投资、禁卖空」下的最大夏普组合（无基数约束）。"""
    from scipy.optimize import minimize
    N = mu.size
    w0 = np.full(N, 1.0 / N)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * N
    res = minimize(_neg_sharpe_w, w0, args=(mu, Sigma, r_f), method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.maximum(res.x, 0.0)
    w /= (w.sum() + _EPS)
    return _portfolio_stats(w, mu, Sigma, r_f)


def equal_weight(mu: np.ndarray, Sigma: np.ndarray, r_f: float = 0.0) -> dict:
    w = np.full(mu.size, 1.0 / mu.size)
    return _portfolio_stats(w, mu, Sigma, r_f)


def _portfolio_stats(w, mu, Sigma, r_f) -> dict:
    ret = float(w @ mu)
    var = float(w @ Sigma @ w)
    sharpe = (ret - r_f) / np.sqrt(max(var, _EPS))
    return {"sharpe": sharpe, "neg_sharpe": -sharpe, "ret": ret,
            "variance": var, "cardinality": int((w > 1e-6).sum()), "weights": w}


# ---------------------------------------------------------------- 聚类
def kmeans_pp(X: np.ndarray, K: int, n_init: int = 10, seed: int = 0) -> dict:
    """k-means++ 多次重启，返回最优 SSE 与（可选）外部指标。"""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, init="k-means++", n_init=n_init, random_state=seed)
    labels = km.fit_predict(X)
    return {"sse": float(km.inertia_), "labels": labels, "centers": km.cluster_centers_}
