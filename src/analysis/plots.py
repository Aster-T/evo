"""出图：箱线图、收敛曲线、变化恢复曲线、参数敏感性。

所有图输出到 figures/（pdf 矢量 + png 预览）。坐标轴标签用英文以避免中文字体缺失；
图题/说明可在报告中用中文撰写。

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

from src.experiments.config import ALGORITHM_NAMES, INSTANCES, build_config

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
PALETTE = {"GA": "#d62728", "PSO": "#1f77b4", "DE": "#2ca02c"}


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.pdf / .png")


def load_curves(npz: str = "curves.npz") -> dict:
    data = np.load(RESULTS_DIR / npz)
    out = {}
    for k in data.files:
        kind, key = k.split("::", 1)
        algo, inst, seed = key.split("|")
        out.setdefault((kind, inst, algo), []).append(data[k])
    return out


# ---------------------------------------------------------------- #
def fig_boxplot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = list(INSTANCES.keys())
    sns.boxplot(data=df, x="instance", y="offline_error", hue="algorithm",
                order=order, hue_order=ALGORITHM_NAMES, palette=PALETTE, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Problem instance")
    ax.set_ylabel("Offline Error (log scale)")
    ax.set_title("Offline Error across instances (15 runs each)")
    ax.legend(title="Algorithm")
    _save(fig, "fig2_boxplot")


def fig_convergence(curves: dict, instance: str = "F1") -> None:
    cfg = build_config(instance)
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ALGORITHM_NAMES:
        arrs = curves.get(("curve", instance, algo))
        if not arrs:
            continue
        M = np.vstack(arrs)
        mean = M.mean(0)
        x = np.linspace(0, cfg.max_evals, len(mean))
        ax.plot(x, mean, label=algo, color=PALETTE[algo], lw=1.8)
        ax.fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0),
                        color=PALETTE[algo], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Current best error (log scale)")
    ax.set_title(f"Convergence on {instance} (mean ± std over runs)")
    ax.legend(title="Algorithm")
    _save(fig, "fig1_convergence")


def fig_recovery(curves: dict, instance: str = "F8", n_changes: int = 6) -> None:
    """放大若干次环境变化前后的误差曲线，展示变化后的恢复过程（锯齿）。"""
    cfg = build_config(instance)
    fig, ax = plt.subplots(figsize=(9, 5))
    npoints = None
    for algo in ALGORITHM_NAMES:
        arrs = curves.get(("curve", instance, algo))
        if not arrs:
            continue
        M = np.vstack(arrs)
        mean = M.mean(0)
        npoints = len(mean)
        x = np.linspace(0, cfg.max_evals, npoints)
        ax.plot(x, mean, label=algo, color=PALETTE[algo], lw=1.5)
    # 只看前 n_changes 个环境
    xmax = cfg.change_frequency * n_changes
    ax.set_xlim(0, xmax)
    for e in range(1, n_changes):
        ax.axvline(e * cfg.change_frequency, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_xlabel("Function evaluations (dashed = environment change)")
    ax.set_ylabel("Current best error")
    ax.set_title(f"Recovery after environmental changes on {instance}")
    ax.legend(title="Algorithm")
    _save(fig, "fig3_recovery")


def fig_sensitivity(seeds=(0, 1, 2, 3, 4), instance="F4", env_number=30) -> None:
    """参数敏感性：DE 的 F 与 PSO 的子群数 n_swarms。需现跑小规模实验。"""
    from src.algorithms import DE, PSO
    from src.gmpb import Benchmark

    def run(algo_cls, params, inst, seed):
        cfg = build_config(inst, env_number)
        b = Benchmark(cfg, seed=seed)
        rng = np.random.default_rng(seed + 10_000)
        algo_cls(cfg.dim, cfg.min_coordinate, cfg.max_coordinate, rng, **params).run(b)
        return b.offline_error

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # DE: scaling factor F
    F_vals = [0.3, 0.5, 0.7, 0.9]
    de_mean, de_std = [], []
    for F in F_vals:
        vals = [run(DE, dict(pop_size=50, F=F, CR=0.9, restart_fraction=0.3), instance, s) for s in seeds]
        de_mean.append(np.mean(vals)); de_std.append(np.std(vals))
    axes[0].errorbar(F_vals, de_mean, yerr=de_std, marker="o", color=PALETTE["DE"], capsize=3)
    axes[0].set_xlabel("DE scaling factor F"); axes[0].set_ylabel("Offline Error")
    axes[0].set_title(f"DE sensitivity to F ({instance})")

    # PSO: number of swarms
    sw_vals = [1, 3, 5, 10]
    pso_mean, pso_std = [], []
    for m in sw_vals:
        vals = [run(PSO, dict(n_swarms=m, swarm_size=max(2, 50 // m)), instance, s) for s in seeds]
        pso_mean.append(np.mean(vals)); pso_std.append(np.std(vals))
    axes[1].errorbar(sw_vals, pso_mean, yerr=pso_std, marker="s", color=PALETTE["PSO"], capsize=3)
    axes[1].set_xlabel("PSO number of swarms"); axes[1].set_ylabel("Offline Error")
    axes[1].set_title(f"PSO sensitivity to #swarms ({instance})")

    fig.suptitle("Parameter sensitivity analysis")
    _save(fig, "fig4_sensitivity")


def main():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    suffix = "raw.csv" if (RESULTS_DIR / "raw.csv").exists() else "raw_quick.csv"
    npz = "curves.npz" if (RESULTS_DIR / "curves.npz").exists() else "curves_quick.npz"
    df = pd.read_csv(RESULTS_DIR / suffix)
    curves = load_curves(npz)

    print("生成图表 →")
    fig_boxplot(df)
    conv_inst = "F1" if ("curve", "F1", "GA") in curves else df.instance.iloc[0]
    fig_convergence(curves, conv_inst)
    rec_inst = "F8" if ("curve", "F8", "GA") in curves else conv_inst
    fig_recovery(curves, rec_inst)
    fig_sensitivity()
    print("完成。")


if __name__ == "__main__":
    main()
