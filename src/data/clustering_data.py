"""聚类数据：UCI Wine Quality（真实表格数据）。

红/白葡萄酒理化指标（11 维连续特征）+ 质量评分（0–10 整数标签）。
来源：Cortez et al. (2009)，UCI ML Repository。
我们做 z-score 标准化（簇中心优化在统一尺度下进行），并把质量评分作为
「真实标签」用于事后计算 ARI / NMI（聚类质量的外部指标）。

用法（下载/缓存）：
  conda run -n evo python -m src.data.clustering_data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WINE_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "wine-quality/winequality-{color}.csv")


def _download_wine(color: str) -> Path:
    """下载 winequality-{color}.csv 到 data/（已存在则跳过）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / f"winequality-{color}.csv"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    import requests
    resp = requests.get(WINE_URL.format(color=color), timeout=30)
    resp.raise_for_status()
    dst.write_text(resp.text, encoding="utf-8")
    return dst


def load_wine(color: str = "red") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """加载 Wine Quality → (X_std, labels, feature_names)。

    X_std  —— z-score 标准化后的特征 (n, 11)
    labels —— 质量评分（整数，作为外部聚类评价的真值）
    """
    path = _download_wine(color)
    df = pd.read_csv(path, sep=";")
    y = df["quality"].to_numpy()
    X = df.drop(columns=["quality"]).to_numpy(dtype=float)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    feature_names = list(df.drop(columns=["quality"]).columns)
    return X_std, y, feature_names


def main() -> None:
    print("下载/缓存 UCI Wine Quality …")
    for color in ("red", "white"):
        X, y, feats = load_wine(color)
        print(f"  {color:5s}: X={X.shape}  特征={len(feats)}  "
              f"质量评分∈[{y.min()},{y.max()}]  类别数={len(np.unique(y))}")


if __name__ == "__main__":
    main()
