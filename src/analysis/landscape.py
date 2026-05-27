"""GMPB landscape 可视化（2D）。

在 dim=2 的 GMPB 上画等高线 + 峰中心，并对比"环境变化前后"两个相邻环境，
直观展示动态优化中地形的移动。用于报告 §1.4。

用法： conda run -n evo python -m src.analysis.landscape
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.gmpb import Benchmark, GMPBConfig

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def _grid_fitness(bench: Benchmark, peaks, n=300):
    lo, hi = bench.cfg.min_coordinate, bench.cfg.max_coordinate
    xs = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z = bench._fitness(pts, peaks).reshape(n, n)
    return X, Y, Z


def main():
    FIG_DIR.mkdir(exist_ok=True)
    cfg = GMPBConfig(dim=2, peak_number=7, shift_severity=5.0,
                     change_frequency=1000, environment_number=3)
    bench = Benchmark(cfg, seed=3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, env_idx, title in zip(axes, [0, 1], ["Environment t", "Environment t+1 (after change)"]):
        p = bench.envs[env_idx]
        X, Y, Z = _grid_fitness(bench, p)
        cs = ax.contourf(X, Y, Z, levels=40, cmap="viridis")
        ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.3, alpha=0.4)
        ax.scatter(p.positions[:, 0], p.positions[:, 1], c="red", marker="x", s=70,
                   label="peak centers")
        gi = int(np.argmax(p.heights))
        ax.scatter(p.positions[gi, 0], p.positions[gi, 1], facecolors="none",
                   edgecolors="white", s=200, lw=2, label="global optimum")
        ax.set_title(title)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(cs, ax=ax, shrink=0.85, label="fitness")

    fig.suptitle("GMPB landscape (2-D) before and after an environmental change")
    fig.savefig(FIG_DIR / "fig0_landscape.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig0_landscape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figures/fig0_landscape.pdf / .png  (峰移动幅度 shift={cfg.shift_severity})")


if __name__ == "__main__":
    main()
