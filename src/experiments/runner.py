"""单次实验运行：给定 (算法, 实例, 种子)，构造基准与算法并运行，返回指标与曲线。

公平性保证：同一 (实例, 种子) 下，基准的环境序列完全相同；算法各自的随机性由
同一 seed 派生的独立 Generator 控制。
"""
from __future__ import annotations

import numpy as np

from src.algorithms import ALGORITHMS
from src.gmpb import Benchmark

from .config import ALGO_PARAMS, build_config
from .metrics import downsample_curve, environment_means, recovery_speed


def run_one(algorithm: str, instance: str, seed: int,
            environment_number: int = 50, curve_points: int = 1000) -> dict:
    cfg = build_config(instance, environment_number)
    # 基准用 seed 决定环境序列；算法用派生 seed，避免与基准随机流耦合
    bench = Benchmark(cfg, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    algo = ALGORITHMS[algorithm](cfg.dim, cfg.min_coordinate, cfg.max_coordinate,
                                 rng, **ALGO_PARAMS[algorithm])
    algo.run(bench)

    eh = bench.error_history
    metrics = {
        "algorithm": algorithm,
        "instance": instance,
        "seed": seed,
        "dim": cfg.dim,
        "peak_number": cfg.peak_number,
        "change_frequency": cfg.change_frequency,
        "shift_severity": cfg.shift_severity,
        "offline_error": bench.offline_error,
        "bbc_error": bench.best_error_before_change,
        "recovery_speed": recovery_speed(eh, cfg.change_frequency, environment_number),
        "max_evals": bench.max_evals,
    }
    curve = {
        "algorithm": algorithm,
        "instance": instance,
        "seed": seed,
        "curve": downsample_curve(eh, curve_points),
        "env_means": environment_means(eh, cfg.change_frequency, environment_number),
    }
    return {"metrics": metrics, "curve": curve}
