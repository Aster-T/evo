"""GMPB 基准参数定义。

默认值取自官方 InitializingPeaks.m，并对应 GMPB 的 "完整" 设定
（EllipticalPeaks=1、含旋转与不规则性变换，类似官方 scenario 4/8）。
课程实例只需覆盖 peak_number / dim / shift_severity / change_frequency 四项，
其余保持官方默认，即可获得保真的 GMPB 地形。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GMPBConfig:
    # —— 实例区分维度（课程 6 个实例据此变化）——
    dim: int = 5
    peak_number: int = 10
    shift_severity: float = 1.0
    change_frequency: int = 5000
    environment_number: int = 50

    # —— 搜索域 ——
    min_coordinate: float = -50.0
    max_coordinate: float = 50.0

    # —— 峰高 ——
    min_height: float = 30.0
    max_height: float = 70.0
    height_severity: float = 7.0

    # —— 峰宽 ——
    min_width: float = 1.0
    max_width: float = 12.0
    width_severity: float = 1.0

    # —— 旋转角（驱动 Givens 旋转的标量角，逐峰随环境漂移）——
    min_angle: float = -math.pi
    max_angle: float = math.pi
    angle_severity: float = math.pi / 9.0

    # —— 不规则性变换参数 tau / eta ——
    min_tau: float = 0.0
    max_tau: float = 0.4
    tau_severity: float = 0.05
    min_eta: float = 10.0
    max_eta: float = 25.0
    eta_severity: float = 2.0

    # 椭圆峰：True 时每维独立宽度（完整 GMPB）；False 时各维同宽（传统 MPB）
    elliptical_peaks: bool = True

    @property
    def max_evals(self) -> int:
        return self.change_frequency * self.environment_number

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.min_coordinate, self.max_coordinate)
