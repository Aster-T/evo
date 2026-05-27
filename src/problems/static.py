"""静态目标适配层：让现有 GA/PSO/DE 无改动地求解静态实数优化问题。

现有算法主循环 `Optimizer.run(benchmark)` 只依赖 benchmark 的四个接口：
`evaluate(X)` / `done` / `changed` / `acknowledge_change()`（见 src/algorithms/base.py）。
GMPB 的 Benchmark 是动态的；本类则冒充一个**静态**基准：

  * `evaluate(X)`  —— 计 FE、记录 best-so-far 收敛曲线；**最小化问题取负返回**
                      （算法假设越大越好，见 base.py 的 `_update_best` 用 argmax）。
  * `done`         —— FE 达到预算 `max_evals` 后置 True。
  * `changed`      —— 恒为 False ⇒ 算法的 `_react()`（DOP 重启）永不触发。
  * `acknowledge_change()` —— 空操作。

如此即可对同一套算法做「仿真（动态 GMPB）」与「真实数据（静态）」两类实验，公平对比。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.experiments.metrics import downsample_curve


class StaticObjective:
    """把任意静态目标函数包装成「Benchmark 接口」供优化器调用。

    参数
    ----
    func : 可调用，接受 (n, d) 数组、返回 (n,) 原始目标值。
    dim  : 决策变量维度。
    lb, ub : 标量边界（统一各维）。
    max_evals : 函数评估预算。
    maximize : True 表示目标越大越好；False（默认）表示最小化。
               对外暴露的 `best_f` 始终是**原始符号**下的最优目标值。
    """

    changed = False  # 静态问题：永不变化 ⇒ 不触发算法的 _react()

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], dim: int,
                 lb: float, ub: float, max_evals: int, maximize: bool = False):
        self.func = func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_evals = int(max_evals)
        self.maximize = maximize
        self.sign = 1.0 if maximize else -1.0  # 内部统一成「最大化 sign*f」

        self.fe = 0
        self._best_signed = -np.inf  # 内部「越大越好」口径的最优（canonical）
        self.best_x: np.ndarray | None = None
        # 逐 FE 记录 best-so-far（原始目标值），供下采样成收敛曲线
        self._curve = np.empty(self.max_evals, dtype=float)

    @property
    def done(self) -> bool:
        return self.fe >= self.max_evals

    def acknowledge_change(self) -> None:  # 静态问题无变化可确认
        pass

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """评估一批解。X 为 (n, d) 或 (d,)。返回供算法最大化的适应度 = sign * f。"""
        X = np.asarray(X, dtype=float)
        scalar = X.ndim == 1
        X2d = X[None, :] if scalar else X

        raw = np.asarray(self.func(X2d), dtype=float).ravel()  # 原始目标 (n,)

        # —— 记账：FE 计数 + best-so-far 曲线（仅在预算内）——
        # 内部统一成「sign*f 越大越好」，best_f 对外暴露原始符号值。
        for i in range(raw.size):
            if self.fe >= self.max_evals:
                break
            signed = self.sign * raw[i]
            if signed > self._best_signed:
                self._best_signed = signed
                self.best_x = X2d[i].copy()
            self._curve[self.fe] = self.best_f  # best-so-far（原始符号）
            self.fe += 1

        fit = self.sign * raw  # 算法最大化此值
        return float(fit[0]) if scalar else fit

    @property
    def best_f(self) -> float:
        """原始符号下的最优目标值（最小化问题即最小值）。"""
        return self.sign * self._best_signed

    # —— 指标 —— #
    def convergence_curve(self, n_points: int = 1000) -> np.ndarray:
        """best-so-far 目标值曲线（原始符号），下采样到 n_points。"""
        return downsample_curve(self._curve[:self.fe], n_points)
