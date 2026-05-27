"""真实数据统计汇总与显著性检验（复用 stats.py 的通用函数）。

投资组合主指标 neg_sharpe、聚类主指标 sse 均为「越小越好」，与 stats.py 语义一致。
产出（results/）：
  summary_{tag}.csv / wilcoxon_{tag}.csv / friedman_{tag}.txt   （tag = portfolio / clustering）

用法： conda run -n evo python -m src.analysis.stats_real
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.stats import friedman_nemenyi, summary_table, wilcoxon_table
from src.experiments.config_real import CLUSTER_ORDER, PORTFOLIO_ORDER

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _run(tag: str, metric: str, order: list[str]) -> None:
    csv = RESULTS_DIR / f"raw_{tag}.csv"
    if not csv.exists():
        print(f"[{tag}] 缺 raw_{tag}.csv，跳过。")
        return
    df = pd.read_csv(csv)
    present = [i for i in order if i in df.instance.unique()]

    summ = summary_table(df, metric=metric, instance_order=present)
    summ.to_csv(RESULTS_DIR / f"summary_{tag}.csv", index=False)
    wil = wilcoxon_table(df, metric=metric, instance_order=present)
    wil.to_csv(RESULTS_DIR / f"wilcoxon_{tag}.csv", index=False)
    fried = friedman_nemenyi(df, metric=metric, instance_order=present)
    (RESULTS_DIR / f"friedman_{tag}.txt").write_text(fried, encoding="utf-8")

    print(f"\n=== [{tag}] {metric} 汇总（mean/std） ===")
    print(summ[["instance", "algorithm", "mean", "std"]].to_string(index=False))
    print(f"\n=== [{tag}] Wilcoxon 配对检验 ===")
    print(wil[["instance", "pair", "better", "p_value", "significant_0.05"]].to_string(index=False))
    print(f"\n=== [{tag}] Friedman + Nemenyi ===")
    print(fried)
    print(f"\n[{tag}] 已写入 summary_{tag}.csv / wilcoxon_{tag}.csv / friedman_{tag}.txt")


def main() -> None:
    _run("portfolio", "neg_sharpe", PORTFOLIO_ORDER)
    _run("clustering", "sse", CLUSTER_ORDER)


if __name__ == "__main__":
    main()
