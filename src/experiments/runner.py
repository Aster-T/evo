"""单次实验运行：给定 (算法, 实例, 种子) 构造问题与算法并运行，返回指标与曲线。

公平性：同一 (实例, 种子) 下问题完全相同；算法随机性由 seed 派生的独立 Generator 控制；
三算法共享相同 FE 预算（由 ContinuousObjective / TSPProblem 记账）。
"""
from __future__ import annotations

import numpy as np

from src.algorithms.continuous import CONTINUOUS_ALGORITHMS
from src.algorithms.tsp import TSP_ALGORITHMS
from src.benchmarks import FUNCTIONS, ContinuousObjective
from src.tsp import TSPProblem

from .config import (CONT_PARAMS, FUNC_DIM, FUNC_MAX_EVALS, TSP_MAX_EVALS,
                     TSP_PARAMS)

SUCCESS_THRESHOLD = 1e-2  # 连续：best_f 低于此视为“成功找到全局最优区域”


def run_one_function(algorithm: str, function: str, seed: int,
                     dim: int = FUNC_DIM, max_evals: int = FUNC_MAX_EVALS,
                     curve_points: int = 1000) -> dict:
    obj = ContinuousObjective(FUNCTIONS[function], dim=dim, max_evals=max_evals)
    rng = np.random.default_rng(seed + 10_000)
    CONTINUOUS_ALGORITHMS[algorithm](obj, rng, **CONT_PARAMS[algorithm]).run()
    metrics = {
        "algorithm": algorithm, "instance": function, "seed": seed, "dim": dim,
        "best_f": obj.best_f, "success": int(obj.best_f < SUCCESS_THRESHOLD),
        "max_evals": max_evals,
    }
    curve = {"algorithm": algorithm, "instance": function, "seed": seed,
             "curve": obj.convergence_curve(curve_points)}
    return {"metrics": metrics, "curve": curve}


def run_one_tsp(algorithm: str, instance: str, seed: int,
                max_evals: int = TSP_MAX_EVALS, curve_points: int = 1000) -> dict:
    p = TSPProblem(instance, max_evals=max_evals)
    rng = np.random.default_rng(seed + 10_000)
    TSP_ALGORITHMS[algorithm](p, rng, **TSP_PARAMS[algorithm]).run()
    metrics = {
        "algorithm": algorithm, "instance": instance, "seed": seed, "n": p.n,
        "best_len": p.best_len, "gap": p.gap(), "optimum": p.optimum,
        "max_evals": max_evals,
    }
    curve = {"algorithm": algorithm, "instance": instance, "seed": seed,
             "curve": p.convergence_curve(curve_points),
             "best_tour": p.best_tour}
    return {"metrics": metrics, "curve": curve}
