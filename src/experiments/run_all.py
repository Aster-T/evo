"""批量实验入口（仿真测试函数 + 真实 TSP），joblib 并行。

产出（results/）：
  raw_functions.csv / curves_functions.npz —— 连续函数每 run 指标与收敛曲线
  raw_tsp.csv / curves_tsp.npz             —— TSP 每 run 指标、收敛曲线、最优巡回

用法：
  conda run -n evo python -m src.experiments.run_all --problem all
  conda run -n evo python -m src.experiments.run_all --problem all --quick
  conda run -n evo python -m src.experiments.run_all --problem tsp --jobs 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import (ALGORITHM_NAMES, FUNCTION_INSTANCES, FUNC_MAX_EVALS,
                     QUICK_FUNC_EVALS, QUICK_TSP_EVALS, SEEDS, TSP_INSTANCES,
                     TSP_MAX_EVALS)
from .runner import run_one_function, run_one_tsp

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _save_functions(results: list) -> None:
    df = pd.DataFrame([r["metrics"] for r in results])
    df.to_csv(RESULTS_DIR / "raw_functions.csv", index=False)
    curves = {f"curve::{r['curve']['algorithm']}|{r['curve']['instance']}|{r['curve']['seed']}":
              r["curve"]["curve"] for r in results}
    np.savez_compressed(RESULTS_DIR / "curves_functions.npz", **curves)
    print(f"[functions] 保存 {len(results)} runs → raw_functions.csv / curves_functions.npz")


def _save_tsp(results: list) -> None:
    df = pd.DataFrame([r["metrics"] for r in results])
    df.to_csv(RESULTS_DIR / "raw_tsp.csv", index=False)
    arrs = {}
    for r in results:
        c = r["curve"]
        key = f"{c['algorithm']}|{c['instance']}|{c['seed']}"
        arrs[f"curve::{key}"] = c["curve"]
        if c.get("best_tour") is not None:
            arrs[f"tour::{key}"] = c["best_tour"]
    np.savez_compressed(RESULTS_DIR / "curves_tsp.npz", **arrs)
    print(f"[tsp] 保存 {len(results)} runs → raw_tsp.csv / curves_tsp.npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["functions", "tsp", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    seeds = [0, 1] if args.quick else SEEDS

    if args.problem in ("functions", "all"):
        funcs = ["Sphere", "Rastrigin"] if args.quick else FUNCTION_INSTANCES
        ev = QUICK_FUNC_EVALS if args.quick else FUNC_MAX_EVALS
        jobs = [(a, f, s) for a in ALGORITHM_NAMES for f in funcs for s in seeds]
        print(f"[functions] {len(jobs)} runs（evals={ev}）…")
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_one_function)(a, f, s, max_evals=ev) for a, f, s in jobs)
        print(f"[functions] 用时 {time.time()-t0:.1f}s")
        _save_functions(res)

    if args.problem in ("tsp", "all"):
        insts = ["berlin52", "eil51"] if args.quick else TSP_INSTANCES
        ev = QUICK_TSP_EVALS if args.quick else TSP_MAX_EVALS
        jobs = [(a, ins, s) for a in ALGORITHM_NAMES for ins in insts for s in seeds]
        print(f"[tsp] {len(jobs)} runs（evals={ev}）…")
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_one_tsp)(a, ins, s, max_evals=ev) for a, ins, s in jobs)
        print(f"[tsp] 用时 {time.time()-t0:.1f}s")
        _save_tsp(res)


if __name__ == "__main__":
    main()
