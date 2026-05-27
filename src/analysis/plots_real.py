"""真实数据出图：收敛曲线、箱线图、与基线对比、聚类 PCA 散点。

输出到 figures/（pdf 矢量 + png 预览）。坐标轴用英文以避免中文字体缺失。
用法： conda run -n evo python -m src.analysis.plots_real
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.experiments.config_real import (CLUSTER_ORDER, MAX_EVALS,
                                          PORTFOLIO_ORDER)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
PALETTE = {"GA": "#d62728", "PSO": "#1f77b4", "DE": "#2ca02c"}
ALGOS = ["GA", "PSO", "DE"]


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.pdf / .png")


def _load_curves(tag: str) -> dict:
    data = np.load(RESULTS_DIR / f"curves_{tag}.npz")
    out: dict = {}
    for k in data.files:
        _, key = k.split("::", 1)
        algo, inst, seed = key.split("|")
        out.setdefault((inst, algo), []).append(data[k])
    return out


# ---------------------------------------------------------------- 投资组合
def fig_portfolio_box(df: pd.DataFrame, baselines: pd.DataFrame) -> None:
    order = [i for i in PORTFOLIO_ORDER if i in df.instance.unique()]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="instance", y="sharpe", hue="algorithm",
                order=order, hue_order=ALGOS, palette=PALETTE, ax=ax)
    # 叠加凸 QP 最优作为参考线（每个实例一个短横线）
    qp = baselines[baselines.method == "QP_maxSharpe"].set_index("instance")["sharpe"]
    for xi, inst in enumerate(order):
        if inst in qp.index:
            ax.hlines(qp[inst], xi - 0.4, xi + 0.4, colors="black", ls="--", lw=1.4)
    ax.set_xlabel("Portfolio instance")
    ax.set_ylabel("Sharpe ratio (higher is better)")
    ax.set_title("Portfolio Sharpe across instances (dashed = convex-QP optimum)")
    ax.legend(title="Algorithm", loc="best")
    _save(fig, "real_fig1_portfolio_box")


def fig_portfolio_convergence(curves: dict, instance: str, max_evals: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ALGOS:
        arrs = curves.get((instance, algo))
        if not arrs:
            continue
        M = -np.vstack(arrs)  # 存的是 neg_sharpe → 取负得 Sharpe
        mean = M.mean(0)
        x = np.linspace(0, max_evals, len(mean))
        ax.plot(x, mean, label=algo, color=PALETTE[algo], lw=1.8)
        ax.fill_between(x, mean - M.std(0), mean + M.std(0), color=PALETTE[algo], alpha=0.15)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best Sharpe ratio so far")
    ax.set_title(f"Portfolio convergence on {instance} (mean ± std)")
    ax.legend(title="Algorithm")
    _save(fig, "real_fig2_portfolio_convergence")


def fig_portfolio_gap(df: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """各算法最优 Sharpe 与凸 QP / 等权基线对比（分组柱状）。"""
    order = [i for i in PORTFOLIO_ORDER if i in df.instance.unique()]
    best = df.groupby(["instance", "algorithm"])["sharpe"].max().unstack()[ALGOS]
    qp = baselines[baselines.method == "QP_maxSharpe"].set_index("instance")["sharpe"]
    ew = baselines[baselines.method == "EqualWeight"].set_index("instance")["sharpe"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    w = 0.16
    for j, algo in enumerate(ALGOS):
        ax.bar(x + (j - 1.5) * w, [best.loc[i, algo] for i in order], w,
               label=algo, color=PALETTE[algo])
    ax.bar(x + 1.5 * w, [qp.get(i, np.nan) for i in order], w, label="QP optimum", color="black")
    ax.plot(x, [ew.get(i, np.nan) for i in order], "v", color="gray", label="Equal weight")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Sharpe ratio"); ax.set_xlabel("Portfolio instance")
    ax.set_title("Best Sharpe per algorithm vs convex-QP / equal-weight baselines")
    ax.legend(ncol=2, loc="upper left", framealpha=0.9)
    _save(fig, "real_fig3_portfolio_gap")


# ---------------------------------------------------------------- 聚类
def fig_clustering_box(df: pd.DataFrame, baselines: pd.DataFrame) -> None:
    order = [i for i in CLUSTER_ORDER if i in df.instance.unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="instance", y="sse", hue="algorithm",
                order=order, hue_order=ALGOS, palette=PALETTE, ax=ax)
    km = baselines.set_index("instance")["sse"]
    for xi, inst in enumerate(order):
        if inst in km.index:
            ax.hlines(km[inst], xi - 0.4, xi + 0.4, colors="black", ls="--", lw=1.4)
    ax.set_xlabel("Clustering instance")
    ax.set_ylabel("SSE (lower is better)")
    ax.set_title("Clustering SSE across instances (dashed = k-means++ baseline)")
    ax.legend(title="Algorithm")
    _save(fig, "real_fig4_clustering_box")


def fig_clustering_convergence(curves: dict, instance: str, max_evals: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ALGOS:
        arrs = curves.get((instance, algo))
        if not arrs:
            continue
        M = np.vstack(arrs)  # best-so-far SSE
        mean = M.mean(0)
        x = np.linspace(0, max_evals, len(mean))
        ax.plot(x, mean, label=algo, color=PALETTE[algo], lw=1.8)
        ax.fill_between(x, mean - M.std(0), mean + M.std(0), color=PALETTE[algo], alpha=0.15)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best SSE so far")
    ax.set_title(f"Clustering convergence on {instance} (mean ± std)")
    ax.legend(title="Algorithm")
    _save(fig, "real_fig5_clustering_convergence")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    print("生成真实数据图表 →")

    pf = RESULTS_DIR / "raw_portfolio.csv"
    if pf.exists():
        df = pd.read_csv(pf)
        base = pd.read_csv(RESULTS_DIR / "baselines_portfolio.csv")
        curves = _load_curves("portfolio")
        fig_portfolio_box(df, base)
        inst = "PORT1" if "PORT1" in df.instance.values else df.instance.iloc[0]
        fig_portfolio_convergence(curves, inst, df["max_evals"].iloc[0])
        fig_portfolio_gap(df, base)

    cf = RESULTS_DIR / "raw_clustering.csv"
    if cf.exists():
        df = pd.read_csv(cf)
        base = pd.read_csv(RESULTS_DIR / "baselines_clustering.csv")
        curves = _load_curves("clustering")
        fig_clustering_box(df, base)
        inst = "WINE_RED" if "WINE_RED" in df.instance.values else df.instance.iloc[0]
        fig_clustering_convergence(curves, inst, df["max_evals"].iloc[0])
    print("完成。")


if __name__ == "__main__":
    main()
