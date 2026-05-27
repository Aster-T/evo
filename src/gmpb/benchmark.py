"""Benchmark：统一评估接口 + FE 计数 + 误差记账 + 环境推进。

移植自 benchmark_func.m 的语义：
  - 全局最优值 Optimum = max(峰高)（每环境重算）；
  - CurrentError = Optimum - f(x) >= 0；
  - 每个环境内维护 "已找到的最佳误差" 的运行最小值，逐 FE 记入 FitnessHistory；
  - Offline Error = mean(FitnessHistory)。

约定：算法只能通过本类的 evaluate() 访问目标函数，所有指标记账集中在此，
避免算法自报指标。一次 evaluate(X) 视作 len(X) 次函数评估，整批在当前环境内完成；
当累计 FE 跨过 change_frequency 边界时，切换到下一环境并置 changed=True（模仿官方 CtrlFlag=1）。
"""
from __future__ import annotations

import numpy as np

from .config import GMPBConfig
from .generator import generate_environments
from .transform import transform


class Benchmark:
    def __init__(self, cfg: GMPBConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.envs = generate_environments(cfg, seed)
        self.optima = np.array([e.optimum_value for e in self.envs])
        self.max_evals = cfg.max_evals

        # 运行状态
        self.fe = 0
        self.env_idx = 0
        self.done = False
        self.changed = False
        self._best_err_in_env = np.inf
        self.error_history = np.empty(self.max_evals, dtype=float)
        self.bbc_errors: list[float] = []  # 每个环境结束时的最佳误差（E_BBC 用）

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fitness(X: np.ndarray, peaks) -> np.ndarray:
        """向量化适应度：f(x) = max_k [ h_k - sqrt(sum_j (w_{k,j} * T(R_k(x-c_k))_j)^2) ]。"""
        diff = X[:, None, :] - peaks.positions[None, :, :]          # (n, m, d)
        z = np.einsum("kil,nkl->nki", peaks.rotation, diff)         # (n, m, d) = R_k (x-c_k)
        zt = transform(z, peaks.tau, peaks.eta)                     # (n, m, d)
        inside = np.sum((peaks.widths[None, :, :] ** 2) * zt ** 2, axis=2)  # (n, m)
        val = peaks.heights[None, :] - np.sqrt(inside)             # (n, m)
        return val.max(axis=1)                                      # (n,)

    # ------------------------------------------------------------------ #
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """评估解。X 为 (n, d) 或 (d,)；返回 (n,) 或标量。"""
        X = np.asarray(X, dtype=float)
        scalar = X.ndim == 1
        X2d = X[None, :] if scalar else X

        peaks = self.envs[self.env_idx]
        f = self._fitness(X2d, peaks)

        if not self.done:
            self._account(f)

        return float(f[0]) if scalar else f

    def _account(self, f: np.ndarray) -> None:
        n = len(f)
        allowed = self.max_evals - self.fe
        k = min(n, allowed)
        if k <= 0:
            self.done = True
            return

        opt = self.optima[self.env_idx]
        errs = opt - f[:k]
        running = np.minimum.accumulate(errs)
        combined = np.minimum(running, self._best_err_in_env)
        self.error_history[self.fe:self.fe + k] = combined
        self._best_err_in_env = float(combined[-1])
        self.fe += k

        if self.fe >= self.max_evals:
            self.bbc_errors.append(self._best_err_in_env)
            self.done = True
            return

        new_env = self.fe // self.cfg.change_frequency
        if new_env > self.env_idx:
            self.bbc_errors.append(self._best_err_in_env)      # 变化前最佳误差
            self.env_idx = min(new_env, self.cfg.environment_number - 1)
            self._best_err_in_env = np.inf
            self.changed = True

    # ------------------------------------------------------------------ #
    def acknowledge_change(self) -> None:
        """算法响应环境变化后调用，清除 changed 标志。"""
        self.changed = False

    @property
    def offline_error(self) -> float:
        """Offline Error = 已发生 FE 上误差运行最小值的平均。"""
        if self.fe == 0:
            return float("nan")
        return float(self.error_history[: self.fe].mean())

    @property
    def best_error_before_change(self) -> float:
        """E_BBC = 各环境结束时最佳误差的平均。"""
        if not self.bbc_errors:
            return float("nan")
        return float(np.mean(self.bbc_errors))

    @property
    def current_optimum(self) -> float:
        return float(self.optima[self.env_idx])
