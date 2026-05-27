"""真实数据实验配置：投资组合 + 聚类（复用 GMPB 的 GA/PSO/DE 参数）。

实例命名与 GMPB 的 F1–F12 平行，便于统一汇总/绘图：
  投资组合：PORT1–PORT5（Beasley 真实指数）+ SP50（yfinance，自建，可缺失）；
            另含 PORT1F/PORT2F 为**无基数约束**版本，用于演示 EA ≈ 凸 QP 最优。
  聚类：    WINE_RED / WINE_WHITE（UCI Wine Quality 簇中心优化）。
"""
from __future__ import annotations

from src.data.clustering_data import load_wine
from src.data.portfolio_data import fetch_sp500, load_beasley
from src.problems.clustering import ClusteringProblem
from src.problems.portfolio import PortfolioProblem

# 复用 GMPB 的算法超参数与算法集合
from .config import ALGO_PARAMS, ALGORITHM_NAMES  # noqa: F401

N_REPEATS = 15
SEEDS = list(range(N_REPEATS))
MAX_EVALS = 30000
QUICK_MAX_EVALS = 3000

# —— 投资组合实例（K=基数约束；None=无约束） ——
PORTFOLIO_INSTANCES: dict[str, dict] = {
    "PORT1":  dict(source="beasley", file="port1.txt", K=10, market="Hang Seng (31)"),
    "PORT2":  dict(source="beasley", file="port2.txt", K=10, market="DAX 100 (85)"),
    "PORT3":  dict(source="beasley", file="port3.txt", K=10, market="FTSE 100 (89)"),
    "PORT4":  dict(source="beasley", file="port4.txt", K=10, market="S&P 100 (98)"),
    "PORT5":  dict(source="beasley", file="port5.txt", K=10, market="Nikkei 225 (225)"),
    "PORT1F": dict(source="beasley", file="port1.txt", K=None, market="Hang Seng (31), 无约束"),
    "PORT2F": dict(source="beasley", file="port2.txt", K=None, market="DAX 100 (85), 无约束"),
    "SP50":   dict(source="yfinance", n=50, K=10, market="S&P 自建 (≤50)"),
}

# —— 聚类实例（sample_cap：评估时的样本上限，控制单次评估开销） ——
CLUSTER_INSTANCES: dict[str, dict] = {
    "WINE_RED":   dict(color="red",   K=6, sample_cap=None),
    "WINE_WHITE": dict(color="white", K=7, sample_cap=2000),
}

PORTFOLIO_ORDER = list(PORTFOLIO_INSTANCES.keys())
CLUSTER_ORDER = list(CLUSTER_INSTANCES.keys())


def build_portfolio_problem(name: str, max_evals: int = MAX_EVALS) -> PortfolioProblem | None:
    """构造投资组合问题；数据不可得（如 yfinance 失败）返回 None。"""
    spec = PORTFOLIO_INSTANCES[name]
    if spec["source"] == "beasley":
        mu, Sigma, _ = load_beasley(spec["file"])
    else:  # yfinance
        out = fetch_sp500()
        if out is None:
            return None
        mu, Sigma, _ = out
    return PortfolioProblem(mu, Sigma, K=spec["K"], objective="sharpe", max_evals=max_evals)


def build_clustering_problem(name: str, max_evals: int = MAX_EVALS,
                             seed: int = 0) -> ClusteringProblem:
    """构造聚类问题（可按 sample_cap 下采样样本，随机但由 seed 固定）。"""
    spec = CLUSTER_INSTANCES[name]
    X, y, _ = load_wine(spec["color"])
    cap = spec.get("sample_cap")
    if cap is not None and X.shape[0] > cap:
        import numpy as np
        rng = np.random.default_rng(12345)  # 固定子样本，保证所有 run 同一数据
        idx = rng.choice(X.shape[0], cap, replace=False)
        X, y = X[idx], y[idx]
    return ClusteringProblem(X, spec["K"], labels_true=y, max_evals=max_evals)
