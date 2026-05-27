"""真实数据批量实验入口（投资组合 + 聚类），joblib 并行。

产出（results/）：
  raw_portfolio.csv / curves_portfolio.npz   —— 投资组合每 run 指标与收敛曲线
  raw_clustering.csv / curves_clustering.npz —— 聚类每 run 指标与收敛曲线
  baselines_portfolio.csv / baselines_clustering.csv —— 基线参照（凸 QP / 等权 / k-means++）

用法：
  conda run -n evo python -m src.experiments.run_all_real --problem all
  conda run -n evo python -m src.experiments.run_all_real --problem all --quick
  conda run -n evo python -m src.experiments.run_all_real --problem portfolio --jobs 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.analysis.baselines import equal_weight, kmeans_pp, qp_max_sharpe

from .config_real import (ALGORITHM_NAMES, CLUSTER_INSTANCES, MAX_EVALS,
                          PORTFOLIO_INSTANCES, QUICK_MAX_EVALS, SEEDS,
                          build_clustering_problem, build_portfolio_problem)
from .runner_real import run_one_clustering, run_one_portfolio

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _save(results: list, tag: str) -> None:
    """保存一组 run 的指标与曲线（自动剔除被跳过的 None）。"""
    results = [r for r in results if r is not None]
    if not results:
        print(f"[{tag}] 无有效结果（可能数据全部不可得），跳过保存。")
        return
    df = pd.DataFrame([r["metrics"] for r in results])
    df.to_csv(RESULTS_DIR / f"raw_{tag}.csv", index=False)
    curve_dict = {}
    for r in results:
        c = r["curve"]
        curve_dict[f"curve::{c['algorithm']}|{c['instance']}|{c['seed']}"] = c["curve"]
    np.savez_compressed(RESULTS_DIR / f"curves_{tag}.npz", **curve_dict)
    print(f"[{tag}] 已保存 {len(results)} 个 run → raw_{tag}.csv / curves_{tag}.npz")


def _portfolio_baselines(instances: list[str], max_evals: int) -> None:
    rows = []
    for ins in instances:
        prob = build_portfolio_problem(ins, max_evals)
        if prob is None:
            continue
        for name, fn in (("QP_maxSharpe", qp_max_sharpe), ("EqualWeight", equal_weight)):
            s = fn(prob.mu, prob.Sigma)
            rows.append({"instance": ins, "method": name, "sharpe": s["sharpe"],
                         "neg_sharpe": s["neg_sharpe"], "variance": s["variance"],
                         "cardinality": s["cardinality"]})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "baselines_portfolio.csv", index=False)
        print(f"[portfolio] 基线已保存 → baselines_portfolio.csv")


def _clustering_baselines(instances: list[str], max_evals: int) -> None:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows = []
    for ins in instances:
        prob = build_clustering_problem(ins, max_evals)
        b = kmeans_pp(prob.X, prob.K)
        ari = adjusted_rand_score(prob.labels_true, b["labels"]) if prob.labels_true is not None else float("nan")
        nmi = normalized_mutual_info_score(prob.labels_true, b["labels"]) if prob.labels_true is not None else float("nan")
        rows.append({"instance": ins, "method": "KMeans++", "sse": b["sse"],
                     "ari": float(ari), "nmi": float(nmi)})
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "baselines_clustering.csv", index=False)
        print(f"[clustering] 基线已保存 → baselines_clustering.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["portfolio", "clustering", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="小规模冒烟")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--evals", type=int, default=None, help="覆盖 FE 预算")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    max_evals = args.evals or (QUICK_MAX_EVALS if args.quick else MAX_EVALS)
    seeds = [0, 1] if args.quick else SEEDS

    do_port = args.problem in ("portfolio", "all")
    do_clust = args.problem in ("clustering", "all")

    if do_port:
        instances = (["PORT1", "PORT1F", "SP50"] if args.quick
                     else list(PORTFOLIO_INSTANCES.keys()))
        jobs = [(a, ins, s) for a in ALGORITHM_NAMES for ins in instances for s in seeds]
        print(f"[portfolio] {len(jobs)} runs（evals={max_evals}）…")
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_one_portfolio)(a, ins, s, max_evals) for a, ins, s in jobs)
        print(f"[portfolio] 用时 {time.time()-t0:.1f}s")
        _save(res, "portfolio")
        _portfolio_baselines(instances, max_evals)

    if do_clust:
        instances = (["WINE_RED"] if args.quick else list(CLUSTER_INSTANCES.keys()))
        jobs = [(a, ins, s) for a in ALGORITHM_NAMES for ins in instances for s in seeds]
        print(f"[clustering] {len(jobs)} runs（evals={max_evals}）…")
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_one_clustering)(a, ins, s, max_evals) for a, ins, s in jobs)
        print(f"[clustering] 用时 {time.time()-t0:.1f}s")
        _save(res, "clustering")
        _clustering_baselines(instances, max_evals)


if __name__ == "__main__":
    main()
