"""实验配置：测试函数实例 + TSP 实例 + 三算法参数 + 重复次数。

仿真数据（连续）：5 个 Appendix C 函数，统一维度 d=30。
真实数据（组合）：4 个 TSPLIB 真实城市坐标实例（a280 作可选大规模）。
"""
from __future__ import annotations

# —— 重复与预算 ——
N_REPEATS = 15
SEEDS = list(range(N_REPEATS))
FUNC_DIM = 30
FUNC_MAX_EVALS = 30000
TSP_MAX_EVALS = 20000
QUICK_FUNC_EVALS = 5000
QUICK_TSP_EVALS = 5000

# —— 仿真：测试函数实例（名称即 src.benchmarks.FUNCTIONS 的 key）——
FUNCTION_INSTANCES = ["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank"]

# —— 真实：TSP 实例（a280 默认不跑，规模大较慢）——
TSP_INSTANCES = ["berlin52", "eil51", "kroA100", "ch150"]
TSP_INSTANCES_FULL = ["berlin52", "eil51", "kroA100", "ch150", "a280"]

ALGORITHM_NAMES = ["GA", "ACO", "PSO"]

# —— 连续算法参数（src.algorithms.continuous）——
CONT_PARAMS: dict[str, dict] = {
    "GA":  dict(pop_size=50, crossover_prob=0.9, eta_c=15.0, eta_m=20.0),
    "ACO": dict(archive_size=50, n_ants=10, q=1e-4, xi=0.85),   # ACOR
    "PSO": dict(swarm_size=50, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, v_frac=0.2),
}

# —— TSP 算法参数（src.algorithms.tsp）——
TSP_PARAMS: dict[str, dict] = {
    "GA":  dict(pop_size=100, crossover_prob=0.9, mutation_prob=0.2, tournament=3),
    "ACO": dict(alpha=1.0, beta=2.0, rho=0.5, Q=1.0),           # Ant System
    "PSO": dict(swarm_size=50, w=0.3, c1=0.7, c2=0.9),         # 交换序列离散 PSO
}
