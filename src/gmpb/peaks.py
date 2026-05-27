"""峰参数初始化（移植自 InitializingPeaks.m）。

单分量 GMPB（MPBnumber=1, Weight=1）。一个 Peaks 表示一个环境下全部峰的参数。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import GMPBConfig
from .rotation import random_orthogonal


@dataclass
class Peaks:
    """单个环境的全部峰参数。所有数组第 0 轴为峰索引 k=0..m-1。"""

    positions: np.ndarray        # (m, d) 峰中心
    heights: np.ndarray          # (m,)   峰高
    widths: np.ndarray           # (m, d) 每维宽度
    angle: np.ndarray            # (m,)   标量旋转角（驱动 Givens 连乘）
    initial_rotation: np.ndarray  # (m, d, d) 固定的随机正交基
    rotation: np.ndarray         # (m, d, d) 当前环境实际旋转矩阵
    tau: np.ndarray              # (m,)
    eta: np.ndarray              # (m, 4)

    def copy(self) -> "Peaks":
        return replace(
            self,
            positions=self.positions.copy(),
            heights=self.heights.copy(),
            widths=self.widths.copy(),
            angle=self.angle.copy(),
            initial_rotation=self.initial_rotation.copy(),
            rotation=self.rotation.copy(),
            tau=self.tau.copy(),
            eta=self.eta.copy(),
        )

    @property
    def optimum_value(self) -> float:
        """该环境全局最优值 = max(峰高)（Weight=1）。"""
        return float(self.heights.max())


def initialize_peaks(cfg: GMPBConfig, rng: np.random.Generator) -> Peaks:
    m, d = cfg.peak_number, cfg.dim

    # 宽度：椭圆峰每维独立；否则各维同宽
    if cfg.elliptical_peaks:
        widths = cfg.min_width + (cfg.max_width - cfg.min_width) * rng.random((m, d))
    else:
        col = cfg.min_width + (cfg.max_width - cfg.min_width) * rng.random((m, 1))
        widths = np.repeat(col, d, axis=1)

    positions = cfg.min_coordinate + (cfg.max_coordinate - cfg.min_coordinate) * rng.random((m, d))
    heights = cfg.min_height + (cfg.max_height - cfg.min_height) * rng.random(m)

    # 旋转：d>1 且开启旋转时用随机正交基；否则单位阵
    rotate = d > 1 and cfg.angle_severity != 0
    if rotate:
        initial_rotation = np.stack([random_orthogonal(d, rng) for _ in range(m)])
    else:
        initial_rotation = np.broadcast_to(np.eye(d), (m, d, d)).copy()
    rotation = initial_rotation.copy()  # 首环境 RotationMatrix = InitialRotationMatrix

    if rotate:
        angle = cfg.min_angle + (cfg.max_angle - cfg.min_angle) * rng.random(m)
    else:
        angle = np.zeros(m)

    if cfg.eta_severity == 0 and cfg.tau_severity == 0:
        tau = np.zeros(m)
        eta = np.zeros((m, 4))
    else:
        tau = cfg.min_tau + (cfg.max_tau - cfg.min_tau) * rng.random(m)
        eta = cfg.min_eta + (cfg.max_eta - cfg.min_eta) * rng.random((m, 4))

    return Peaks(
        positions=positions,
        heights=heights,
        widths=widths,
        angle=angle,
        initial_rotation=initial_rotation,
        rotation=rotation,
        tau=tau,
        eta=eta,
    )
