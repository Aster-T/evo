"""真实数据单次运行：构造静态问题与算法并运行，返回指标与收敛曲线。

公平性：同一 (实例, 种子) 下问题完全相同；算法随机性由 seed 派生的独立 Generator 控制。
所有算法只通过 StaticObjective.evaluate 访问目标，FE 预算一致。
"""
from __future__ import annotations

import numpy as np

from src.algorithms import ALGORITHMS

from .config_real import (ALGO_PARAMS, MAX_EVALS, build_clustering_problem,
                          build_portfolio_problem)
from .metrics import downsample_curve


def _run(problem, algorithm: str, seed: int):
    """通用：构造算法、跑满 FE 预算，返回 (静态目标, 最优 genotype)。"""
    obj = problem.make_objective()
    rng = np.random.default_rng(seed + 10_000)
    algo = ALGORITHMS[algorithm](obj.dim, obj.lb, obj.ub, rng, **ALGO_PARAMS[algorithm])
    algo.run(obj)
    return obj


def run_one_portfolio(algorithm: str, instance: str, seed: int,
                      max_evals: int = MAX_EVALS, curve_points: int = 1000) -> dict | None:
    problem = build_portfolio_problem(instance, max_evals)
    if problem is None:  # 数据不可得（如 yfinance 失败）→ 跳过
        return None
    obj = _run(problem, algorithm, seed)
    m = problem.metrics_of(obj.best_x)
    metrics = {
        "algorithm": algorithm, "instance": instance, "seed": seed,
        "dim": problem.N, "K": problem.K if problem.K is not None else problem.N,
        "neg_sharpe": m["neg_sharpe"], "sharpe": m["sharpe"],
        "variance": m["variance"], "cardinality": m["cardinality"],
        "max_evals": max_evals,
    }
    curve = {"algorithm": algorithm, "instance": instance, "seed": seed,
             "curve": obj.convergence_curve(curve_points)}
    return {"metrics": metrics, "curve": curve}


def run_one_clustering(algorithm: str, instance: str, seed: int,
                       max_evals: int = MAX_EVALS, curve_points: int = 1000) -> dict | None:
    problem = build_clustering_problem(instance, max_evals, seed)
    obj = _run(problem, algorithm, seed)
    m = problem.metrics_of(obj.best_x)
    metrics = {
        "algorithm": algorithm, "instance": instance, "seed": seed,
        "dim": problem.dim, "K": problem.K,
        "sse": m["sse"], "ari": m.get("ari", float("nan")),
        "nmi": m.get("nmi", float("nan")), "max_evals": max_evals,
    }
    curve = {"algorithm": algorithm, "instance": instance, "seed": seed,
             "curve": obj.convergence_curve(curve_points)}
    return {"metrics": metrics, "curve": curve}
