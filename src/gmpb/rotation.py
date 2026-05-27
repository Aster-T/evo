"""旋转矩阵构造（移植自 Rotation.m / InitializingPeaks.m）。

官方做法：
  - 初始旋转矩阵 = qr(rand(d)) 得到的随机正交矩阵（InitialRotationMatrix）；
  - 每个环境的旋转矩阵 = InitialRotationMatrix @ Rotation(angle, d)，
    其中 Rotation(angle, d) 是用 *同一个标量角* angle 对所有维度对 (i,j), i<j
    的 Givens 旋转连乘得到的正交矩阵。
"""
from __future__ import annotations

import numpy as np


def givens_product(theta: float, dim: int) -> np.ndarray:
    """对所有维度对 (i<j) 以同一角度 theta 做 Givens 旋转并连乘。

    复刻 Rotation.m：output = G_1 G_2 ... G_{d(d-1)/2}，每个 G 形如
        [ii,ii]=cos, [jj,jj]=cos, [ii,jj]=sin, [jj,ii]=-sin。
    """
    out = np.eye(dim)
    c, s = np.cos(theta), np.sin(theta)
    # 右乘 Givens(i,j) 只改变第 i、j 两列，按列更新即可（等价于 out @ G）
    for i in range(dim):
        for j in range(i + 1, dim):
            ci = out[:, i].copy()
            cj = out[:, j].copy()
            out[:, i] = ci * c - cj * s
            out[:, j] = ci * s + cj * c
    return out


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """随机正交矩阵，复刻 MATLAB 的 qr(rand(d))（取 Q）。"""
    q, _ = np.linalg.qr(rng.random((dim, dim)))
    return q
