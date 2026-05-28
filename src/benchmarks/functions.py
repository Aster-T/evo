"""连续测试函数（仿真数据，取自 Dan Simon《Evolutionary Optimization
Algorithms》Appendix C）。

选用 5 个经典函数，覆盖不同地形难度（均为最小化，全局最优 f*=0）：

| 函数 (Appendix C) | 表达式要点 | 定义域(各维) | 特性 |
|----|----|----|----|
| Sphere (C.1.1)     | Σ x_i²                                   | [-5.12, 5.12]   | 单峰、可分、最易 |
| Rosenbrock (C.1.4) | Σ 100(x_{i+1}-x_i²)²+(x_i-1)²            | [-2.048, 2.048] | 香蕉谷、强变量耦合 |
| Rastrigin (C.1.11) | 10d + Σ[x_i²-10cos(2πx_i)]               | [-5.12, 5.12]   | 规则强多峰 |
| Ackley (C.1.2)     | -20e^{-0.2√(Σx²/d)} - e^{Σcos(2πx)/d}+20+e | [-30, 30]     | 多峰、近全局有大量浅峰 |
| Griewank (C.1.6)   | Σx²/4000 - Πcos(x_i/√i) + 1              | [-600, 600]     | 多峰、变量弱耦合 |

每个函数接受 (n, d) 批量输入，返回 (n,) 适应度，便于种群批量评估。
全局最优点：Rosenbrock 在 x=1（全 1），其余在 x=0。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def sphere(X: np.ndarray) -> np.ndarray:
    return np.sum(X ** 2, axis=1)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    a = X[:, :-1]
    b = X[:, 1:]
    return np.sum(100.0 * (b - a ** 2) ** 2 + (a - 1.0) ** 2, axis=1)


def rastrigin(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    return 10.0 * d + np.sum(X ** 2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


def ackley(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    s1 = np.sum(X ** 2, axis=1)
    s2 = np.sum(np.cos(2.0 * np.pi * X), axis=1)
    return (-20.0 * np.exp(-0.2 * np.sqrt(s1 / d))
            - np.exp(s2 / d) + 20.0 + np.e)


def griewank(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    i = np.arange(1, d + 1)
    return (np.sum(X ** 2, axis=1) / 4000.0
            - np.prod(np.cos(X / np.sqrt(i)), axis=1) + 1.0)


@dataclass(frozen=True)
class FunctionSpec:
    """一个测试函数的全部元信息。"""
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    lb: float
    ub: float
    optimum: float = 0.0     # 全局最优值 f*
    section: str = ""        # 教材 Appendix C 小节号


FUNCTIONS: dict[str, FunctionSpec] = {
    "Sphere":     FunctionSpec("Sphere", sphere, -5.12, 5.12, 0.0, "C.1.1"),
    "Rosenbrock": FunctionSpec("Rosenbrock", rosenbrock, -2.048, 2.048, 0.0, "C.1.4"),
    "Rastrigin":  FunctionSpec("Rastrigin", rastrigin, -5.12, 5.12, 0.0, "C.1.11"),
    "Ackley":     FunctionSpec("Ackley", ackley, -30.0, 30.0, 0.0, "C.1.2"),
    "Griewank":   FunctionSpec("Griewank", griewank, -600.0, 600.0, 0.0, "C.1.6"),
}

FUNCTION_NAMES = list(FUNCTIONS.keys())
