"""批量实验入口。

完整实验： 3 算法 × 6 实例 × 15 重复 = 270 runs，joblib 多核并行。
产出：
  results/raw.csv     —— 每个 run 一行指标（offline_error / bbc / recovery 等）
  results/curves.npz  —— 收敛曲线（下采样）与逐环境均值，供画图用

用法：
  conda run -n evo python -m src.experiments.run_all              # 完整实验
  conda run -n evo python -m src.experiments.run_all --quick      # 小规模冒烟
  conda run -n evo python -m src.experiments.run_all --jobs 4 --env 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import ALGORITHM_NAMES, INSTANCES, SEEDS
from .runner import run_one

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def build_jobs(quick: bool):
    if quick:
        algos = ALGORITHM_NAMES
        instances = ["F1", "F4"]
        seeds = [0, 1]
        env_number = 10
    else:
        algos = ALGORITHM_NAMES
        instances = list(INSTANCES.keys())
        seeds = SEEDS
        env_number = 50
    jobs = [(a, ins, s) for a in algos for ins in instances for s in seeds]
    return jobs, env_number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小规模冒烟实验")
    ap.add_argument("--jobs", type=int, default=-1, help="并行进程数（-1=全部核）")
    ap.add_argument("--env", type=int, default=None, help="覆盖环境数")
    args = ap.parse_args()

    jobs, env_number = build_jobs(args.quick)
    if args.env is not None:
        env_number = args.env

    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"运行 {len(jobs)} 个 run（环境数={env_number}，并行={args.jobs}）...")
    t0 = time.time()

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_one)(a, ins, s, env_number) for (a, ins, s) in jobs
    )

    dt = time.time() - t0
    print(f"完成，用时 {dt:.1f}s（{dt / len(jobs):.2f}s/run）")

    # —— 保存指标 ——
    df = pd.DataFrame([r["metrics"] for r in results])
    suffix = "_quick" if args.quick else ""
    csv_path = RESULTS_DIR / f"raw{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"指标已保存：{csv_path}")

    # —— 保存曲线 ——
    curve_dict = {}
    for r in results:
        c = r["curve"]
        key = f"{c['algorithm']}|{c['instance']}|{c['seed']}"
        curve_dict[f"curve::{key}"] = c["curve"]
        curve_dict[f"envmean::{key}"] = c["env_means"]
    npz_path = RESULTS_DIR / f"curves{suffix}.npz"
    np.savez_compressed(npz_path, **curve_dict)
    print(f"曲线已保存：{npz_path}")

    # —— 控制台速览 ——
    summary = df.groupby(["instance", "algorithm"])["offline_error"].mean().unstack()
    print("\nOffline Error 均值（行=实例，列=算法）：")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
