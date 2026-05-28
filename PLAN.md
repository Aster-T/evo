# 仿生算法课程作业 — 实验计划（v3：GA / ACO / PSO）

> **课程**：2025-2026《仿生算法的数学基础》
> **作业**：数据分析报告（满分 50 分），截止 2026-06-15
> **提交**：PDF → 班级联络人潘文豪 → 邮箱 2694196312@qq.com

---

## 一、研究方向

**题目**：基于 GA / ACO / PSO 三种仿生算法的优化对比分析（连续测试函数 + 真实 TSP）

**三算法**：遗传算法 GA、蚁群算法 ACO、粒子群算法 PSO。

**两类数据**：
1. **仿真数据**：教材（Dan Simon《Evolutionary Optimization Algorithms》）**Appendix C** 的 5 个
   经典连续测试函数：Sphere、Rosenbrock、Rastrigin、Ackley、Griewank（覆盖单峰/谷型/强多峰/多峰）。
2. **真实数据**：**TSPLIB 真实城市坐标 TSP**（berlin52/eil51/kroA100/ch150；a280 可选）——
   真实地理坐标、自带已知最优解，是 ACO 的本职基准。

**为何这样配**：GA/PSO 天然连续，ACO 本为组合优化（TSP）而生。故 TSP 用经典蚁群、
连续函数用其标准连续推广 **ACOR**（Socha&Dorigo 2008），使三算法在两类问题上都能公平参与。

---

## 二、问题与数学化

- 仿真：`min f(x), x∈[lb,ub]^d`，d=30。定义域（教材 Appendix C）：Sphere ±5.12 / Rosenbrock ±2.048 /
  Rastrigin ±5.12 / Ackley ±30 / Griewank ±600；全局最优 f\*=0。
- 真实：`min Σ d(π_i,π_{i+1}) + d(π_n,π_1)`，π 为城市访问排列；EUC_2D 距离 `round(√(Δx²+Δy²))`。

## 三、算法设计（两种编码形态）

| 算法 | 连续（实数编码） | TSP（排列编码） |
|----|----|----|
| GA  | 锦标赛 + SBX + 多项式变异 + 精英 | 锦标赛 + 顺序交叉 OX + 反转变异 + 精英 |
| ACO | ACOR：解档案 k + 高斯核逐维采样 | Ant System：信息素 τ + η=1/d，概率构造 + 蒸发 + 沉积 |
| PSO | 标准 PSO：惯性权重 + 个体/全局最优 + 速度钳制 | 交换序列离散 PSO（速度=交换算子序列） |

## 四、实验流程

1. **数据准备**：`src.tsp.data` 下载/缓存 TSPLIB；测试函数内置于 `src.benchmarks.functions`。
2. **统一评测接口**：`ContinuousObjective` / `TSPProblem` 负责 FE 计数、best-so-far 曲线、预算判定；
   三算法只调用其 `evaluate`，共享相同 FE 预算（连续 30000 / TSP 20000），保证公平。
3. **批量实验**：`src.experiments.run_all`（joblib 并行）
   - 仿真：3 算法 × 5 函数 × 15 重复（d=30）= 225 runs
   - 真实：3 算法 × 4 TSP 实例 × 15 重复 = 180 runs
4. **统计分析**：`src.analysis.stats` —— 各实例×算法 mean/std；实例内 Wilcoxon；跨实例 Friedman+Nemenyi。
5. **可视化**：`src.analysis.plots` —— 收敛曲线、箱线图、TSP 最优路线图。
6. **报告**：`report/report.md`，按课程附件1大纲。

## 五、评价指标
- 仿真：最终最优 f（与 f\*=0 之差）、成功率（f<1e-2）、收敛曲线。
- 真实：最优巡回长度、**gap%=(找到−已知最优)/已知最优**、收敛曲线、最优路线图。

## 六、目录与命令
见 [CLAUDE.md](CLAUDE.md)。一键复现：
```bash
conda run -n evo python -m src.tsp.data
conda run -n evo python -m src.experiments.run_all --problem all
conda run -n evo python -m src.analysis.stats
conda run -n evo python -m src.analysis.plots
conda run -n evo pytest tests/ -q
```

## 七、关键结论（已完成）
- 仿真：ACOR 光滑地形（Sphere/Rosenbrock）极强但强多峰（Rastrigin/Ackley）早熟最差；
  PSO 连续多峰最好；GA 最稳健（Rastrigin 最优）。Friedman 秩 PSO1.6<ACO2.0<GA2.4（p=0.45）。
- 真实 TSP：**ACO 绝对占优**（gap 0.9–8%），GA 居中，离散 PSO 最弱；Friedman p=0.018，ACO≻PSO 显著。
- **No Free Lunch**：算法选型须匹配问题的编码类型与地形特性。
