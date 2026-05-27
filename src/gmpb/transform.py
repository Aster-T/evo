"""不规则性变换 T(z, tau, eta)（移植自 benchmark_func.m 内的 Transform）。

逐元素作用，制造非对称、不规则（多局部起伏）的峰形：
    z>0 :  exp( ln(z)  + tau*(sin(eta1*ln(z))  + sin(eta2*ln(z)))  )
    z<0 : -exp( ln(-z) + tau*(sin(eta3*ln(-z)) + sin(eta4*ln(-z))) )
    z==0:  0

当 tau=0 且 eta=0 时，T 退化为恒等映射；且对任意 tau/eta 恒有 T(0)=0，
从而保证峰中心 (z=0) 处 f = 峰高，全局最优值 = max(峰高)。
"""
from __future__ import annotations

import numpy as np


def transform(z: np.ndarray, tau: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """对旋转平移后的坐标施加不规则性变换。

    Parameters
    ----------
    z : (..., m, d) 旋转平移后的坐标，最后两轴为 (峰索引, 维度)。
    tau : (m,) 每个峰的 tau。
    eta : (m, 4) 每个峰的 4 个 eta 频率参数。

    Returns
    -------
    (..., m, d) 变换后的坐标。
    """
    z = np.asarray(z, dtype=float)
    tau = np.asarray(tau, dtype=float)
    eta = np.asarray(eta, dtype=float)

    # 将逐峰参数广播到 z 的形状：tau (m,)->(...,m,1)，eta[:,k] (m,)->(...,m,1)
    tau_b = tau[..., :, None]
    eta1 = eta[..., :, 0, None]
    eta2 = eta[..., :, 1, None]
    eta3 = eta[..., :, 2, None]
    eta4 = eta[..., :, 3, None]

    out = np.zeros_like(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_abs = np.log(np.abs(z))  # z=0 处为 -inf，下面用掩码丢弃
        pos = np.exp(ln_abs + tau_b * (np.sin(eta1 * ln_abs) + np.sin(eta2 * ln_abs)))
        neg = -np.exp(ln_abs + tau_b * (np.sin(eta3 * ln_abs) + np.sin(eta4 * ln_abs)))

    out = np.where(z > 0, pos, out)
    out = np.where(z < 0, neg, out)
    return out
