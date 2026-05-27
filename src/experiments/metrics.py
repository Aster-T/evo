"""评价指标。

主指标 Offline Error 与 E_BBC 由 Benchmark 直接维护；本模块补充从误差曲线
派生的辅助指标（恢复速度、收敛曲线下采样）。
"""
from __future__ import annotations

import numpy as np


def downsample_curve(error_history: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """将逐 FE 的误差曲线下采样到 n_points，便于存储与画收敛曲线。"""
    L = len(error_history)
    if L <= n_points:
        return error_history.copy()
    idx = np.linspace(0, L - 1, n_points).astype(int)
    return error_history[idx]


def recovery_speed(error_history: np.ndarray, change_frequency: int,
                   environment_number: int, threshold_ratio: float = 0.2) -> float:
    """平均恢复速度：每次环境变化后，误差从变化时刻值下降到其 threshold_ratio
    所需的 FE 数（按各环境平均；未达到则记为整个环境长度）。值越小恢复越快。"""
    speeds = []
    for e in range(environment_number):
        seg = error_history[e * change_frequency:(e + 1) * change_frequency]
        if len(seg) == 0:
            continue
        start = seg[0]
        target = start * threshold_ratio
        below = np.where(seg <= target)[0]
        speeds.append(int(below[0]) if len(below) else len(seg))
    return float(np.mean(speeds)) if speeds else float("nan")


def environment_means(error_history: np.ndarray, change_frequency: int,
                      environment_number: int) -> np.ndarray:
    """每个环境段内误差均值（用于观察跨环境稳定性）。"""
    out = []
    for e in range(environment_number):
        seg = error_history[e * change_frequency:(e + 1) * change_frequency]
        if len(seg):
            out.append(seg.mean())
    return np.array(out)
