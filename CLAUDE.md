# CLAUDE.md — 项目核心备忘

> 给 Claude（和我自己）随时查看的项目导航与约定。详细计划见 [PLAN.md](PLAN.md) 与
> `~/.claude/plans/plan-md-src-quiet-hickey.md`。

## 项目目标
《仿生算法的数学基础》课程作业：在 **GMPB 动态优化基准** 上对比 **GA / PSO / DE** 三种进化算法。
产出一份数据分析报告（PDF，截止 **2026-06-15**）。本仓库负责实验代码与图表生成。

## 运行环境（重要）
- **所有 Python 操作必须在 conda 环境 `evo`（Python 3.12）下执行。**
- 运行命令统一用：`conda run -n evo python ...` 或 `conda run -n evo pytest ...`
- 依赖见 [requirements.txt](requirements.txt)：numpy / scipy / matplotlib / pandas / seaborn / joblib / scikit-posthocs / pytest。
- 安装的版本较新：numpy 2.x、pandas 3.x、scipy 1.17——注意新版 API。

## 目录结构
```
src/
  gmpb/        GMPB benchmark 移植（peaks/rotation/transform/dynamics/generator/benchmark）
  algorithms/  base + ga + pso + de（均针对 DOP：变化检测 + 重启）
  experiments/ config（6 实例 + 参数）/ metrics / runner / run_all
  analysis/    stats（汇总表+Wilcoxon）/ plots / landscape
  validate/    cross_check（自洽性 + 与 MATLAB 对拍）
tests/         pytest 单元测试
results/       实验原始 csv / npz（git 忽略）
figures/       输出图表 pdf/png（git 忽略）
```

## GMPB 数学定义（移植自 EMI-Group/GMPB，Yazdani 2020）
对解 x（维度 d），适应度取各峰最大值：
```
f(x) = max_{k=1..m} [ h_k - sqrt( sum_j w_{k,j} * z'_{k,j}^2 ) ]
  z_k  = R_k (x - c_k)            # 平移 + 旋转
  z'_k = T(z_k, τ_k, η_k)         # 逐元素不规则性变换
```
不规则性变换 T（制造非对称/不规则地形）：
```
y>0 :  exp( ln(y)  + τ*(sin(η1·ln(y))  + sin(η2·ln(y)))  )
y<0 : -exp( ln(-y) + τ*(sin(η3·ln(-y)) + sin(η4·ln(-y))) )
y==0:  0
```
- 全局最优值 = `max(heights)`（峰中心处 sqrt 项为 0）→ Offline Error 用它做基准。
- 环境变化：每 `ChangeFrequency` 次评估，高度/宽度/角度/τ/η 加高斯扰动并反射到边界；中心沿单位随机方向移动 `ShiftSeverity`；重建旋转矩阵。
- **全部环境在 generator 中预生成并缓存**，保证可复现。

## 核心约定
- **算法只通过 `Benchmark.evaluate(x)` 访问目标函数**；FE 计数、误差记账、环境推进都集中在 Benchmark 内，算法不得自报指标。
- 随机性统一用 `numpy.random.Generator`（`np.random.default_rng(seed)`），便于复现。
- 搜索域默认 `[-50, 50]^d`，高度 `[30,70]`，宽度 `[1,12]`，角度 `[-π,π]`，τ∈[-1,1]、η∈[-1,1]（具体见 config）。

## 问题实例（6 个，src/experiments/config.py）
| 实例 | PeakNumber | Dim | ShiftSeverity | ChangeFreq | 特征 |
|----|----|----|----|----|----|
| F1 | 5 | 5 | 1 | 5000 | 简单基线 |
| F4 | 50 | 5 | 1 | 5000 | 多峰挑战 |
| F8 | 10 | 5 | 1 | 500 | 快速变化 |
| F10| 10 | 10| 1 | 5000 | 中维度 |
| F11| 10 | 20| 1 | 5000 | 高维度 |
| F12| 10 | 5 | 5 | 5000 | 大幅变化 |
`EnvironmentNumber` 默认 50（预实验后可调，控制总时长）。

## 评价指标（src/experiments/metrics.py）
- **Offline Error**（主指标）：所有 FE 上 `(全局最优 - 当前环境内已找到的最佳)` 的平均。
- **E_BBC**：每个环境变化前一刻误差的平均。
- 辅助：恢复速度、稳定性（重复实验 std）、收敛曲线。

## 实验规模
`3 算法 × 6 实例 × 15 重复 = 270 runs`，joblib 并行。`run_all.py --quick` 做小规模冒烟。

## 常用命令
```bash
conda run -n evo pytest tests/ -q                       # 单元测试
conda run -n evo python -m src.validate.cross_check     # 自洽性/对拍验证
conda run -n evo python -m src.experiments.run_all --quick   # 冒烟
conda run -n evo python -m src.experiments.run_all      # 完整实验 → results/
conda run -n evo python -m src.analysis.plots           # 出图 → figures/
conda run -n evo python -m src.analysis.stats           # 汇总表 + Wilcoxon
```

## 进度
- [x] conda 环境 evo + 依赖
- [x] gmpb benchmark（移植自官方 EMI-Group/GMPB，公式逐项核对）
- [x] 单元测试（26 passed）+ 自洽性验证（V1–V7 全通过）
- [x] GA/PSO/DE（均含变化响应；smoke 全部优于随机搜索）
- [x] 实验框架 + 冒烟
- [x] 分析出图（fig0 landscape / fig1 收敛 / fig2 箱线 / fig3 恢复 / fig4 敏感性）
- [x] 完整实验：270 runs，41s 完成 → results/raw.csv 等

## 关键结论（完整实验，Offline Error 均值）
| 实例 | GA | PSO | DE | 最优 |
|----|----|----|----|----|
| F1 (基线) | 14.4 | **3.3** | 18.8 | PSO |
| F4 (50峰) | 17.2 | **7.5** | 21.3 | PSO |
| F8 (快变) | 26.3 | **12.4** | 22.0 | PSO |
| F10 (10维) | 25.6 | **21.0** | 21.7 | PSO≈DE |
| F11 (20维) | 66.0 | 81.3 | **27.2** | DE |
| F12 (大幅移动) | 19.7 | **11.4** | 28.9 | PSO |

- **PSO（多种群+量子粒子）**在低-中维多峰/快变/大幅移动上全面占优；
- **DE/rand/1/bin**在高维(F11)显著最好，且方差最小，鲁棒；PSO 在 20 维偶发发散(max≈379)；
- 与 CEC 文献一致：PSO 变体在 GMPB 上强，但高维需要 DE 类的差分缩放。
- 显著性见 results/wilcoxon.csv（多数 p<0.05）与 results/friedman.txt（PSO 平均秩最优 1.33）。
