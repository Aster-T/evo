"""统计汇总与显著性检验。

产出（写入 results/）：
  summary.csv     —— 表1：各实例×算法 Offline Error 的 best/worst/mean/median/std
  wilcoxon.csv    —— 表2：每个实例内三算法两两 Wilcoxon 符号秩检验 p 值（配对，按种子）
  friedman.txt    —— 跨实例 Friedman 检验 + Nemenyi 事后比较（平均秩）
并在控制台打印 Markdown 表格，便于直接粘进报告。

用法： conda run -n evo python -m src.analysis.stats
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from src.experiments.config import ALGORITHM_NAMES, INSTANCES

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
INSTANCE_ORDER = list(INSTANCES.keys())


def load(metric_csv: str = "raw.csv") -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / metric_csv)


def summary_table(df: pd.DataFrame, metric: str = "offline_error",
                  instance_order: list[str] | None = None) -> pd.DataFrame:
    order = instance_order or INSTANCE_ORDER
    g = df.groupby(["instance", "algorithm"])[metric]
    out = g.agg(best="min", worst="max", mean="mean", median="median", std="std").reset_index()
    out["instance"] = pd.Categorical(out["instance"], order, ordered=True)
    out["algorithm"] = pd.Categorical(out["algorithm"], ALGORITHM_NAMES, ordered=True)
    return out.sort_values(["instance", "algorithm"]).reset_index(drop=True)


def wilcoxon_table(df: pd.DataFrame, metric: str = "offline_error",
                   instance_order: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for inst in (instance_order or INSTANCE_ORDER):
        sub = df[df.instance == inst]
        if sub.empty:
            continue
        # 按 seed 对齐成配对样本
        piv = sub.pivot(index="seed", columns="algorithm", values=metric)
        for a, b in itertools.combinations(ALGORITHM_NAMES, 2):
            if a not in piv.columns or b not in piv.columns:
                continue
            x, y = piv[a].to_numpy(), piv[b].to_numpy()
            if np.allclose(x, y):
                p = 1.0
            else:
                try:
                    p = wilcoxon(x, y).pvalue
                except ValueError:
                    p = float("nan")
            better = a if x.mean() < y.mean() else b  # 误差小者更优
            rows.append({"instance": inst, "pair": f"{a} vs {b}",
                         "mean_left": x.mean(), "mean_right": y.mean(),
                         "better": better, "p_value": p,
                         "significant_0.05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def friedman_nemenyi(df: pd.DataFrame, metric: str = "offline_error",
                     instance_order: list[str] | None = None) -> str:
    """跨实例 Friedman：以每个实例的算法均值为一个 block。"""
    order = instance_order or INSTANCE_ORDER
    means = df.groupby(["instance", "algorithm"])[metric].mean().unstack()[ALGORITHM_NAMES]
    present = [i for i in order if i in means.index]
    means = means.loc[present]
    stat, p = friedmanchisquare(*[means[a].to_numpy() for a in ALGORITHM_NAMES])
    # 平均秩（秩 1 = 最优 = 误差最小）
    ranks = means.rank(axis=1, method="average")
    avg_rank = ranks.mean(axis=0)

    lines = [
        f"跨实例 Friedman 检验（基于各实例 {metric} 均值）",
        f"  统计量 chi2 = {stat:.4f},  p = {p:.4g}",
        "  平均秩（越小越好）：",
    ]
    for a in ALGORITHM_NAMES:
        lines.append(f"    {a:4s}: {avg_rank[a]:.3f}")

    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(means[ALGORITHM_NAMES].to_numpy())
        nem.index = nem.columns = ALGORITHM_NAMES
        lines.append("\n  Nemenyi 事后检验 p 值矩阵：")
        lines.append(nem.round(4).to_string())
    except Exception as e:  # pragma: no cover
        lines.append(f"  (Nemenyi 事后检验跳过：{e})")
    return "\n".join(lines)


def _to_markdown(summary: pd.DataFrame, metric="offline_error") -> str:
    lines = ["| 实例 | 算法 | best | worst | mean | median | std |",
             "|------|------|------|-------|------|--------|-----|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['instance']} | {r['algorithm']} | {r['best']:.3f} | "
                     f"{r['worst']:.3f} | {r['mean']:.3f} | {r['median']:.3f} | {r['std']:.3f} |")
    return "\n".join(lines)


def main(metric_csv: str = "raw.csv"):
    df = load(metric_csv)
    RESULTS_DIR.mkdir(exist_ok=True)

    summ = summary_table(df)
    summ.to_csv(RESULTS_DIR / "summary.csv", index=False)
    wil = wilcoxon_table(df)
    wil.to_csv(RESULTS_DIR / "wilcoxon.csv", index=False)
    fried = friedman_nemenyi(df)
    (RESULTS_DIR / "friedman.txt").write_text(fried, encoding="utf-8")

    print("=== 表1：Offline Error 汇总（Markdown） ===")
    print(_to_markdown(summ))
    print("\n=== 表2：Wilcoxon 配对检验 ===")
    print(wil.to_string(index=False))
    print("\n=== " + "Friedman + Nemenyi" + " ===")
    print(fried)
    print(f"\n已写入 {RESULTS_DIR}/summary.csv, wilcoxon.csv, friedman.txt")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "raw.csv")
