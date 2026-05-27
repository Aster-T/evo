"""实验实例与算法参数集中配置。

6 个问题实例覆盖 4 个差异维度：峰数量、维度、变化幅度、变化频率。
其余 GMPB 参数（高度/宽度/角度/不规则性范围与 severity）取官方默认（见 GMPBConfig）。
"""
from __future__ import annotations

from src.gmpb import GMPBConfig

# 统一环境数与重复次数
ENVIRONMENT_NUMBER = 50
N_REPEATS = 15
SEEDS = list(range(N_REPEATS))

# 6 个问题实例：仅覆盖区分性参数，其余用 GMPBConfig 默认（完整 GMPB 地形）
INSTANCES: dict[str, dict] = {
    "F1":  dict(peak_number=5,  dim=5,  shift_severity=1.0, change_frequency=5000),  # 简单基线
    "F4":  dict(peak_number=50, dim=5,  shift_severity=1.0, change_frequency=5000),  # 多峰挑战
    "F8":  dict(peak_number=10, dim=5,  shift_severity=1.0, change_frequency=500),   # 快速变化
    "F10": dict(peak_number=10, dim=10, shift_severity=1.0, change_frequency=5000),  # 中维度
    "F11": dict(peak_number=10, dim=20, shift_severity=1.0, change_frequency=5000),  # 高维度
    "F12": dict(peak_number=10, dim=5,  shift_severity=5.0, change_frequency=5000),  # 大幅变化
}

INSTANCE_NOTES = {
    "F1": "简单基线", "F4": "多峰挑战", "F8": "快速变化",
    "F10": "中等维度", "F11": "高维度", "F12": "大幅变化",
}

# 算法参数：统一种群 50；区分统一参数与算法专属参数
ALGO_PARAMS: dict[str, dict] = {
    "GA":  dict(pop_size=50, crossover_prob=0.9, eta_c=15.0, eta_m=20.0,
                restart_fraction=0.5),
    "PSO": dict(n_swarms=5, swarm_size=10, chi=0.7298, c1=2.05, c2=2.05,
                r_cloud=1.0, quantum_fraction=0.5),
    "DE":  dict(pop_size=50, F=0.5, CR=0.9, restart_fraction=0.3),
}

ALGORITHM_NAMES = ["GA", "PSO", "DE"]


def build_config(instance_name: str, environment_number: int = ENVIRONMENT_NUMBER) -> GMPBConfig:
    """根据实例名构造完整的 GMPBConfig。"""
    spec = INSTANCES[instance_name]
    return GMPBConfig(environment_number=environment_number, **spec)
