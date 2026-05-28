# GA / ACO / PSO 仿生算法对比（连续测试函数 + 真实 TSP）

《仿生算法的数学基础》课程作业的实验代码：对比 **遗传算法 GA、蚁群算法 ACO、粒子群 PSO**
三种仿生算法，在两类问题上评测——

- **仿真数据（连续优化）**：教材 Dan Simon《Evolutionary Optimization Algorithms》**Appendix C**
  的 5 个经典测试函数（Sphere / Rosenbrock / Rastrigin / Ackley / Griewank）。
- **真实数据（组合优化）**：**TSPLIB 真实城市坐标 TSP**（berlin52 / eil51 / kroA100 / ch150），自带已知最优解。

ACO 本为组合优化而生：TSP 用经典 **Ant System**，连续函数用其标准连续推广 **ACOR**
（Socha & Dorigo 2008），使三算法在两类问题上都能公平参与。详见 [CLAUDE.md](CLAUDE.md) 与 [PLAN.md](PLAN.md)。

## 环境

所有 Python 操作在 conda 环境 `evo`（Python 3.12）下进行：

```bash
conda create -y -n evo python=3.12
conda run -n evo pip install -r requirements.txt
```

## 目录

```
src/
  benchmarks/   连续测试函数（Appendix C）: functions + objective（FE 计数/曲线）
  tsp/          真实数据: data（TSPLIB 下载/解析 + 已知最优）+ problem（距离矩阵/巡回评估）
  algorithms/
    continuous/ GA(实数) + PSO(标准) + ACOR(连续蚁群)
    tsp/        GA(排列 OX) + ACO(Ant System) + PSO(交换序列离散)
  experiments/  config / runner / run_all（joblib 并行）
  analysis/     stats（Wilcoxon + Friedman/Nemenyi）/ plots（收敛/箱线/TSP 路线）
tests/          pytest 单元测试（24 项）
results/        实验产物（csv / npz，git 忽略）
figures/        图表 pdf+png（git 忽略）
data/tsplib/    缓存的 .tsp（git 忽略）
report/         report.md（完整报告）
```

## 复现实验

```bash
conda run -n evo pytest tests/ -q                                   # 单元测试
conda run -n evo python -m src.tsp.data                             # 下载/缓存 TSPLIB
conda run -n evo python -m src.experiments.run_all --problem all --quick   # 冒烟
conda run -n evo python -m src.experiments.run_all --problem all           # 完整实验
conda run -n evo python -m src.analysis.stats                       # 汇总表 + 显著性
conda run -n evo python -m src.analysis.plots                       # 图表 → figures/
```

实验规模：仿真 3 算法 × 5 函数 × 15 重复（d=30，225 runs）；真实 3 算法 × 4 TSP 实例 × 15 重复（180 runs）。

## 评价指标

- 仿真：最终最优 f（与全局最优 f\*=0 之差）、成功率（f<1e-2）、收敛曲线。
- 真实：最优巡回长度、**gap% =(找到−已知最优)/已知最优**、收敛曲线、最优路线图。
- 统计：实例内三算法两两 Wilcoxon；跨实例 Friedman + Nemenyi。

## 主要结论

**仿真（best_f 均值，d=30，加粗=最优）**

| 函数 | GA | ACO(R) | PSO |
|----|----|----|----|
| Sphere（单峰） | 3.0e-3 | **7.8e-40** | 6.0e-5 |
| Rosenbrock（谷型） | 44.2 | **20.7** | 27.0 |
| Rastrigin（强多峰） | **21.1** | 125.3 | 57.0 |
| Ackley（多峰） | 0.83 | 9.38 | **0.33** |
| Griewank（多峰） | 0.80 | 0.17 | **0.07** |

**真实 TSP（gap% 均值，加粗=最优）**

| 实例 | GA | ACO | PSO |
|----|----|----|----|
| berlin52 | 46.6 | **0.9** | 175.4 |
| eil51 | 53.6 | **5.2** | 163.7 |
| kroA100 | 203.0 | **8.1** | 452.7 |
| ch150 | 267.6 | **6.5** | 507.2 |

**No Free Lunch**：ACO 在本职的组合问题 TSP 上绝对占优（gap 0.9–8%，Friedman p=0.018，ACO≻PSO 显著）；
其连续推广 ACOR 在光滑地形（Sphere/Rosenbrock）精度极高，但强多峰上早熟最差；PSO 在连续多峰上最好、
离散用于 TSP 时最弱；GA 最稳健。算法选型须匹配问题的编码类型与地形特性。
