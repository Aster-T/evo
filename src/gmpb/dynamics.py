"""环境变化（移植自 EnvironmentalChange.m）。

每次环境切换，对每个峰的中心/宽度/高度/eta/tau/角度施加高斯扰动，并在越界时
按官方"反射"规则折回边界内；随后用漂移后的角度重建旋转矩阵。
"""
from __future__ import annotations

import numpy as np

from .config import GMPBConfig
from .peaks import Peaks
from .rotation import givens_product


def _reflect(val: np.ndarray, off: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """官方单次反射边界处理：cand=val+off，越下界 → 2lo-val-off，越上界 → 2hi-val-off。"""
    cand = val + off
    out = cand.copy()
    below = cand < lo
    above = cand > hi
    out[below] = 2 * lo - val[below] - off[below]
    out[above] = 2 * hi - val[above] - off[above]
    return out


def environmental_change(peaks: Peaks, cfg: GMPBConfig, rng: np.random.Generator) -> Peaks:
    """返回下一个环境的峰参数（不修改输入）。"""
    m, d = cfg.peak_number, cfg.dim
    p = peaks.copy()

    # —— 峰中心：沿单位随机方向移动 shift_severity ——
    r = rng.standard_normal((m, d))
    norms = np.linalg.norm(r, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    shift = cfg.shift_severity * (r / norms)
    p.positions = _reflect(p.positions, shift, cfg.min_coordinate, cfg.max_coordinate)

    # —— 峰宽 ——
    if cfg.elliptical_peaks:
        w_off = cfg.width_severity * rng.standard_normal((m, d))
    else:
        w_off = np.repeat(cfg.width_severity * rng.standard_normal((m, 1)), d, axis=1)
    p.widths = _reflect(p.widths, w_off, cfg.min_width, cfg.max_width)

    # —— 峰高 ——
    h_off = cfg.height_severity * rng.standard_normal(m)
    p.heights = _reflect(p.heights, h_off, cfg.min_height, cfg.max_height)

    # —— eta / tau ——
    if cfg.eta_severity != 0 or cfg.tau_severity != 0:
        eta_off = cfg.eta_severity * rng.standard_normal((m, 4))
        p.eta = _reflect(p.eta, eta_off, cfg.min_eta, cfg.max_eta)
        tau_off = cfg.tau_severity * rng.standard_normal(m)
        p.tau = _reflect(p.tau, tau_off, cfg.min_tau, cfg.max_tau)

    # —— 旋转角 + 重建旋转矩阵 ——
    if d > 1 and cfg.angle_severity != 0:
        a_off = cfg.angle_severity * rng.standard_normal(m)
        p.angle = _reflect(p.angle, a_off, cfg.min_angle, cfg.max_angle)
        for k in range(m):
            p.rotation[k] = p.initial_rotation[k] @ givens_product(float(p.angle[k]), d)

    return p
