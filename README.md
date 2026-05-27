# GA / PSO / DE 在 GMPB 动态优化基准上的对比

《仿生算法的数学基础》课程作业的实验代码：在 **GMPB（Generalized Moving Peaks
Benchmark, Yazdani 2020 / IEEE CEC 2025）** 动态优化基准上对比遗传算法（GA）、
粒子群（PSO，多种群 mQSO 简化版）与差分进化（DE/rand/1/bin）。

GMPB benchmark 由官方 MATLAB 源码 [EMI-Group/GMPB](https://github.com/EMI-Group/GMPB)
逐项移植为纯 numpy（单分量、Weight=1）。详见 [CLAUDE.md](CLAUDE.md) 与 [PLAN.md](PLAN.md)。

## 环境

所有 Python 操作在 conda 环境 `evo`（Python 3.12）下进行：

```bash
conda create -y -n evo python=3.12
conda run -n evo pip install -r requirements.txt
```

## 目录

```
src/
  gmpb/        GMPB 基准：peaks / rotation / transform / dynamics / generator / benchmark
  algorithms/  base + ga + pso + de（均含环境变化响应：重评 + 部分重启）
  experiments/ config（6 实例 + 参数）/ metrics / runner / run_all
  analysis/    stats（汇总表 + Wilcoxon + Friedman/Nemenyi）/ plots / landscape
  validate/    cross_check（结构性自洽验证 V1–V7）
tests/         pytest 单元测试（26 项）
results/       实验产物（csv / npz，git 忽略）
figures/       图表 pdf+png（git 忽略）
```

## 复现实验

```bash
# 1. 单元测试 + 基准自洽性验证
conda run -n evo pytest tests/ -q
conda run -n evo python -m src.validate.cross_check

# 2. 冒烟（12 runs，~2s）/ 完整实验（270 runs，~40s）
conda run -n evo python -m src.experiments.run_all --quick
conda run -n evo python -m src.experiments.run_all

# 3. 统计与出图
conda run -n evo python -m src.analysis.stats        # → results/summary.csv, wilcoxon.csv, friedman.txt
conda run -n evo python -m src.analysis.plots        # → figures/fig1–4
conda run -n evo python -m src.analysis.landscape    # → figures/fig0
```

## 评价指标

- **Offline Error**（主指标）：所有函数评估上「全局最优 − 当前环境内已找到最佳」的平均；
- **E_BBC**：每个环境变化前一刻最佳误差的平均；
- 辅助：恢复速度、逐环境误差均值、稳定性（重复实验标准差）。

实验规模：3 算法 × 6 实例 × 15 重复 = **270 runs**，joblib 多核并行。

## 主要结论

| 实例 | 特征 | GA | PSO | DE |
|----|----|----|----|----|
| F1 | 简单基线 | 14.4 | **3.3** | 18.8 |
| F4 | 50 峰 | 17.2 | **7.5** | 21.3 |
| F8 | 快速变化 | 26.3 | **12.4** | 22.0 |
| F10 | 10 维 | 25.6 | **21.0** | 21.7 |
| F11 | 20 维 | 66.0 | 81.3 | **27.2** |
| F12 | 大幅移动 | 19.7 | **11.4** | 28.9 |

（Offline Error 均值，越小越好；**加粗**为该实例最优。）

PSO 的多种群 + 量子粒子机制在低-中维多峰与快变环境上全面领先；DE 在高维（F11）
显著最优且方差最小。结论与 CEC GMPB 竞赛中 PSO 变体强势、但高维需差分缩放的认知一致。
