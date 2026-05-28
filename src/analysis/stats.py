"""统计汇总与显著性检验（通用，越小越好）。

对仿真（best_f）与真实 TSP（gap%）分别产出：
  summary_{tag}.csv  —— 各实例×算法 best/worst/mean/median/std
  wilcoxon_{tag}.csv —— 实例内三算法两两 Wilcoxon 配对检验
  friedman_{tag}.txt —— 跨实例 Friedman + Nemenyi 事后比较

用法： conda run -n evo python -m src.analysis.stats
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from src.algorithms import ALGORITHM_NAMES
from src.experiments.config import FUNCTION_INSTANCES, TSP_INSTANCES

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def summary_table(df: pd.DataFrame, metric: str, order: list[str]) -> pd.DataFrame:
    g = df.groupby(["instance", "algorithm"])[metric]
    out = g.agg(best="min", worst="max", mean="mean", median="median", std="std").reset_index()
    out["instance"] = pd.Categorical(out["instance"], order, ordered=True)
    out["algorithm"] = pd.Categorical(out["algorithm"], ALGORITHM_NAMES, ordered=True)
    return out.sort_values(["instance", "algorithm"]).reset_index(drop=True)


def wilcoxon_table(df: pd.DataFrame, metric: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for inst in order:
        sub = df[df.instance == inst]
        if sub.empty:
            continue
        piv = sub.pivot(index="seed", columns="algorithm", values=metric)
        for a, b in itertools.combinations(ALGORITHM_NAMES, 2):
            if a not in piv or b not in piv:
                continue
            x, y = piv[a].to_numpy(), piv[b].to_numpy()
            if np.allclose(x, y):
                p = 1.0
            else:
                try:
                    p = wilcoxon(x, y).pvalue
                except ValueError:
                    p = float("nan")
            better = a if x.mean() < y.mean() else b
            rows.append({"instance": inst, "pair": f"{a} vs {b}",
                         "mean_left": x.mean(), "mean_right": y.mean(),
                         "better": better, "p_value": p,
                         "significant_0.05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def friedman_nemenyi(df: pd.DataFrame, metric: str, order: list[str]) -> str:
    means = df.groupby(["instance", "algorithm"])[metric].mean().unstack()[ALGORITHM_NAMES]
    present = [i for i in order if i in means.index]
    means = means.loc[present]
    stat, p = friedmanchisquare(*[means[a].to_numpy() for a in ALGORITHM_NAMES])
    ranks = means.rank(axis=1, method="average")
    avg_rank = ranks.mean(axis=0)
    lines = [f"跨实例 Friedman 检验（基于各实例 {metric} 均值）",
             f"  统计量 chi2 = {stat:.4f},  p = {p:.4g}",
             "  平均秩（越小越好）："]
    for a in ALGORITHM_NAMES:
        lines.append(f"    {a:4s}: {avg_rank[a]:.3f}")
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(means[ALGORITHM_NAMES].to_numpy())
        nem.index = nem.columns = ALGORITHM_NAMES
        lines.append("\n  Nemenyi 事后检验 p 值矩阵：")
        lines.append(nem.round(4).to_string())
    except Exception as e:  # pragma: no cover
        lines.append(f"  (Nemenyi 跳过：{e})")
    return "\n".join(lines)


def _run(tag: str, metric: str, order: list[str]) -> None:
    csv = RESULTS_DIR / f"raw_{tag}.csv"
    if not csv.exists():
        print(f"[{tag}] 缺 raw_{tag}.csv，跳过。")
        return
    df = pd.read_csv(csv)
    present = [i for i in order if i in df.instance.unique()]
    summary_table(df, metric, present).to_csv(RESULTS_DIR / f"summary_{tag}.csv", index=False)
    wilcoxon_table(df, metric, present).to_csv(RESULTS_DIR / f"wilcoxon_{tag}.csv", index=False)
    fried = friedman_nemenyi(df, metric, present)
    (RESULTS_DIR / f"friedman_{tag}.txt").write_text(fried, encoding="utf-8")

    print(f"\n=== [{tag}] {metric} 均值（行=实例，列=算法）===")
    print(df.groupby(["instance", "algorithm"])[metric].mean().unstack()[ALGORITHM_NAMES].round(4).to_string())
    print(f"\n=== [{tag}] Friedman + Nemenyi ===")
    print(fried)
    print(f"[{tag}] 已写入 summary/wilcoxon/friedman_{tag}.*")


def main() -> None:
    _run("functions", "best_f", FUNCTION_INSTANCES)
    _run("tsp", "gap", TSP_INSTANCES)


if __name__ == "__main__":
    main()
