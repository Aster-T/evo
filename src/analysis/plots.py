"""出图：收敛曲线、箱线图、TSP 最优路线图。

输出到 figures/（pdf 矢量 + png 预览）。坐标轴英文，避免中文字体缺失。
用法： conda run -n evo python -m src.analysis.plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.algorithms import ALGORITHM_NAMES
from src.experiments.config import (FUNCTION_INSTANCES, FUNC_MAX_EVALS,
                                    TSP_INSTANCES, TSP_MAX_EVALS)
from src.tsp import load_coords

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
PALETTE = {"GA": "#d62728", "ACO": "#1f77b4", "PSO": "#2ca02c"}


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
        kind, key = k.split("::", 1)
        algo, inst, seed = key.split("|")
        out.setdefault((kind, inst, algo), []).append(data[k])
    return out


# ---------------------------------------------------------------- 仿真函数
def fig_func_convergence(curves: dict) -> None:
    fig, axes = plt.subplots(1, len(FUNCTION_INSTANCES), figsize=(4 * len(FUNCTION_INSTANCES), 4))
    for ax, fname in zip(axes, FUNCTION_INSTANCES):
        for algo in ALGORITHM_NAMES:
            arrs = curves.get(("curve", fname, algo))
            if not arrs:
                continue
            M = np.vstack(arrs)
            mean = M.mean(0)
            x = np.linspace(0, FUNC_MAX_EVALS, len(mean))
            ax.plot(x, np.maximum(mean, 1e-12), label=algo, color=PALETTE[algo], lw=1.6)
        ax.set_yscale("log"); ax.set_title(fname)
        ax.set_xlabel("FE"); ax.set_ylabel("best f (log)")
    axes[0].legend(title="Algorithm")
    fig.suptitle("Convergence on Appendix C functions (mean over runs, d=30)")
    _save(fig, "fig1_func_convergence")


def fig_func_box(df: pd.DataFrame) -> None:
    d = df.copy()
    d["best_f"] = d["best_f"].clip(lower=1e-12)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=d, x="instance", y="best_f", hue="algorithm",
                order=FUNCTION_INSTANCES, hue_order=ALGORITHM_NAMES, palette=PALETTE, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Test function"); ax.set_ylabel("Final best f (log scale)")
    ax.set_title("Final objective across functions (15 runs each)")
    ax.legend(title="Algorithm")
    _save(fig, "fig2_func_box")


# ---------------------------------------------------------------- TSP
def fig_tsp_box(df: pd.DataFrame) -> None:
    order = [i for i in TSP_INSTANCES if i in df.instance.unique()]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="instance", y="gap", hue="algorithm",
                order=order, hue_order=ALGORITHM_NAMES, palette=PALETTE, ax=ax)
    ax.set_xlabel("TSP instance"); ax.set_ylabel("Gap to optimum (%)")
    ax.set_title("TSP optimality gap across instances (15 runs each)")
    ax.legend(title="Algorithm")
    _save(fig, "fig3_tsp_box")


def fig_tsp_convergence(curves: dict, instance: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ALGORITHM_NAMES:
        arrs = curves.get(("curve", instance, algo))
        if not arrs:
            continue
        M = np.vstack(arrs); mean = M.mean(0)
        x = np.linspace(0, TSP_MAX_EVALS, len(mean))
        ax.plot(x, mean, label=algo, color=PALETTE[algo], lw=1.8)
        ax.fill_between(x, mean - M.std(0), mean + M.std(0), color=PALETTE[algo], alpha=0.15)
    ax.set_xlabel("Tour evaluations"); ax.set_ylabel("Best tour length")
    ax.set_title(f"TSP convergence on {instance} (mean ± std)")
    ax.legend(title="Algorithm")
    _save(fig, "fig4_tsp_convergence")


def fig_tsp_routes(curves: dict, instance: str) -> None:
    """各算法在该实例上的最优巡回路线图（取各算法第一个种子的最优巡回）。"""
    coords = load_coords(instance)
    fig, axes = plt.subplots(1, len(ALGORITHM_NAMES), figsize=(4 * len(ALGORITHM_NAMES), 4))
    for ax, algo in zip(axes, ALGORITHM_NAMES):
        tours = curves.get(("tour", instance, algo))
        ax.scatter(coords[:, 0], coords[:, 1], s=12, c="gray", zorder=2)
        if tours:
            t = tours[0]
            loop = np.append(t, t[0])
            ax.plot(coords[loop, 0], coords[loop, 1], "-", color=PALETTE[algo], lw=1.0, zorder=1)
        ax.set_title(f"{algo}"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Best tours on {instance}")
    _save(fig, "fig5_tsp_routes")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    print("生成图表 →")
    ff = RESULTS_DIR / "raw_functions.csv"
    if ff.exists():
        fig_func_convergence(_load_curves("functions"))
        fig_func_box(pd.read_csv(ff))
    tf = RESULTS_DIR / "raw_tsp.csv"
    if tf.exists():
        curves = _load_curves("tsp")
        fig_tsp_box(pd.read_csv(tf))
        inst = "kroA100" if ("curve", "kroA100", "ACO") in curves else TSP_INSTANCES[0]
        fig_tsp_convergence(curves, inst)
        fig_tsp_routes(curves, "berlin52")
    print("完成。")


if __name__ == "__main__":
    main()
