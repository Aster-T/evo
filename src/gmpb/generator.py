"""预生成全部环境的峰参数（对应 BenchmarkGenerator.m 的组装思路）。

将所有 EnvironmentNumber 个环境一次性生成并缓存为列表，保证：
  1. 给定 seed 完全可复现；
  2. 同一实例下不同算法面对完全相同的环境序列（公平对比）；
  3. 评估时只需按 FE 推进环境索引，无需在线演化。
"""
from __future__ import annotations

import numpy as np

from .config import GMPBConfig
from .dynamics import environmental_change
from .peaks import Peaks, initialize_peaks


def generate_environments(cfg: GMPBConfig, seed: int) -> list[Peaks]:
    """返回长度为 environment_number 的 Peaks 列表（含初始环境）。"""
    rng = np.random.default_rng(seed)
    envs: list[Peaks] = [initialize_peaks(cfg, rng)]
    for _ in range(cfg.environment_number - 1):
        envs.append(environmental_change(envs[-1], cfg, rng))
    return envs
