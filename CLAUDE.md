# CLAUDE.md — 项目核心备忘

> 给 Claude（和我自己）随时查看的项目导航与约定。

## 项目目标
《仿生算法的数学基础》课程作业：对比 **GA / ACO / PSO** 三种仿生算法，分别在
**仿真数据**（教材 Appendix C 连续测试函数）与**真实数据**（TSPLIB 真实城市坐标 TSP）上评测。
产出一份数据分析报告（PDF，截止 **2026-06-15**）。本仓库负责实验代码与图表生成。

## 运行环境（重要）
- **所有 Python 操作必须在 conda 环境 `evo`（Python 3.12）下执行。**
- 命令统一用 `conda run -n evo python ...`；若 `conda run` 异常可直接用
  `D:/miniconda3/envs/evo/python.exe ...`（等价、更稳）。
- 依赖见 [requirements.txt](requirements.txt)：numpy / scipy / matplotlib / pandas / seaborn /
  joblib / scikit-posthocs / requests / pytest。numpy 2.x、pandas 3.x、scipy 1.17，注意新版 API。

## 目录结构
```
src/
  benchmarks/   连续测试函数（Appendix C）: functions（5 函数）+ objective（FE 计数/曲线）
  tsp/          真实数据: data（TSPLIB 下载/解析 + 已知最优）+ problem（距离矩阵/巡回评估）
  algorithms/
    continuous/ GA(实数) + PSO(标准) + ACOR(连续蚁群)
    tsp/        GA(排列OX) + ACO(Ant System) + PSO(交换序列离散)
    common.py   两个轻量基类（按 FE 预算的主循环）
  experiments/  config（实例+参数+seeds）/ runner / run_all（joblib 并行）
  analysis/     stats（Wilcoxon+Friedman+Nemenyi）/ plots（收敛/箱线/TSP路线）
tests/          pytest 单元测试（24 passed）
results/        实验原始 csv / npz（git 忽略）
figures/        输出图表 pdf/png（git 忽略）
data/tsplib/    缓存的 .tsp（git 忽略）
report/         report.md（完整报告，按附件1大纲）
```

## 核心设计约定
- **两类问题、两种编码**：连续（实数向量）与组合（排列）。同一算法思想在两边各有形态：
  ACO 连续=ACOR（Socha&Dorigo 2008 解档案高斯采样），TSP=经典 Ant System。
- 算法只通过 `ContinuousObjective.evaluate` / `TSPProblem.evaluate` 访问目标；**FE 计数、
  best-so-far 曲线、预算判定集中在问题对象内**，算法不自报指标。三算法共享相同 FE 预算（公平）。
- 随机性统一用 `np.random.default_rng(seed)`；同一 (实例,种子) 下问题完全相同。
- 测试函数定义域（教材 Appendix C）：Sphere ±5.12 / Rosenbrock ±2.048 / Rastrigin ±5.12 /
  Ackley ±30 / Griewank ±600；全局最优 f\*=0（Rosenbrock 在 x=1，余在 x=0）。

## 实例与规模
- 仿真：5 函数 × 3 算法 × 15 重复，d=30，FE=30000（225 runs）。
- 真实：TSP {berlin52, eil51, kroA100, ch150}（a280 可选）× 3 算法 × 15 重复，FE=20000（180 runs）。
- 评价：仿真 best_f / 成功率(f<1e-2) / 收敛曲线；TSP 最优巡回长度 / gap% / 收敛 / 路线图。
- 统计：实例内 Wilcoxon；跨实例 Friedman + Nemenyi。

## 常用命令
```bash
conda run -n evo pytest tests/ -q                                  # 单元测试
conda run -n evo python -m src.tsp.data                            # 下载/缓存 TSPLIB
conda run -n evo python -m src.experiments.run_all --problem all --quick   # 冒烟
conda run -n evo python -m src.experiments.run_all --problem all           # 完整 → results/
conda run -n evo python -m src.analysis.stats                      # 汇总表 + 显著性
conda run -n evo python -m src.analysis.plots                      # 出图 → figures/
```

## 进度
- [x] 删除旧 GMPB/DE/投资组合/聚类代码，src/ 全新重写
- [x] 连续测试函数（5 个 Appendix C）+ 目标封装
- [x] TSPLIB 真实数据加载（berlin52/eil51/kroA100/ch150/a280）
- [x] 6 个算法（连续 GA/ACOR/PSO + TSP GA/ACO/PSO）
- [x] 实验框架 + joblib 并行 + 统计 + 出图
- [x] 单元测试 24 passed
- [x] 完整实验：225（函数）+ 180（TSP）runs；report/report.md 已填结果

## 关键结论（完整实验）
**仿真（best_f 均值，d=30）**：Sphere 上 ACOR≈7.8e-40 远超(GA 3e-3/PSO 6e-5)；Rosenbrock ACOR 最优(20.7)；
Rastrigin GA 最优(21.1, ACOR 最差 125)；Ackley/Griewank PSO 最优。Friedman 秩 PSO1.6<ACO2.0<GA2.4(p=0.45)。
→ ACOR 光滑地形极强但强多峰早熟；PSO 多峰最好；GA 最稳。

**真实 TSP（gap% 均值）**：**ACO 绝对占优**（berlin52 0.9% / eil51 5.2% / kroA100 8.1% / ch150 6.5%）；
GA 居中(47–268%)；离散 PSO 最弱(164–507%)。Friedman p=0.018，秩 ACO1.0<GA2.0<PSO3.0，Nemenyi ACO≻PSO p=0.013。

→ **No Free Lunch**：ACO 在本职 TSP 上无敌、PSO 在连续多峰强、GA 全能稳健；算法选型须匹配问题编码与地形。
