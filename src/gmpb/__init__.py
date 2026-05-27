"""GMPB benchmark 的纯 numpy 移植。

移植自官方 MATLAB 源码 EMI-Group/GMPB (Yazdani et al., IEEE TCYB 2020)。
采用单分量（single-component）GMPB：MPBnumber=1，Weight=1，因此全局最优值 = max(峰高)。

公开接口：
    GMPBConfig          —— 基准参数（dataclass）
    Peaks               —— 单个环境的峰参数容器
    Benchmark           —— 评估 + FE 计数 + 误差记账 + 环境推进
    transform           —— 不规则性变换 T(z, tau, eta)
"""
from .config import GMPBConfig
from .peaks import Peaks, initialize_peaks
from .transform import transform
from .rotation import givens_product, random_orthogonal
from .dynamics import environmental_change
from .generator import generate_environments
from .benchmark import Benchmark

__all__ = [
    "GMPBConfig",
    "Peaks",
    "initialize_peaks",
    "transform",
    "givens_product",
    "random_orthogonal",
    "environmental_change",
    "generate_environments",
    "Benchmark",
]
