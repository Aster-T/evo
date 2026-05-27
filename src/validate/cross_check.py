"""GMPB Python 移植的正确性验证。

由于 MATLAB 与 numpy 的随机数流不同，无法靠"同种子复现同数值"做逐点对拍；
因此采用 **结构性自洽验证**（这些性质是 GMPB 数学定义的直接推论，与 RNG 无关）：

  (V1) 峰中心处 f(c_k) == h_k        —— 变换在 0 处为 0；
  (V2) 全局最优值 == max(峰高)        —— Weight=1 的单分量 GMPB；
  (V3) 任意点 f(x) <= 全局最优值       —— f = max_k(h_k - 非负) ;
  (V4) 旋转矩阵正交                    —— R R^T = I；
  (V5) tau=eta=0 时变换为恒等         —— 退化为传统 MPB；
  (V6) CurrentError = Optimum - f >= 0；
  (V7) 给定 seed，环境序列完全可复现。

若本地装有 Octave，可用官方 .m 在相同峰参数下对比（见 README 的"对拍"小节，本脚本默认不依赖）。

用法：  conda run -n evo python -m src.validate.cross_check
"""
from __future__ import annotations

import numpy as np

from src.gmpb import Benchmark, GMPBConfig, transform


def run_checks() -> bool:
    rng = np.random.default_rng(2024)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))

    # 覆盖多种配置：低/高维、含/不含旋转与不规则性
    configs = [
        GMPBConfig(dim=2, peak_number=5, change_frequency=100, environment_number=3),
        GMPBConfig(dim=5, peak_number=10, change_frequency=100, environment_number=3),
        GMPBConfig(dim=10, peak_number=10, change_frequency=100, environment_number=3),
        GMPBConfig(dim=5, peak_number=10, change_frequency=100, environment_number=3,
                   angle_severity=0.0, tau_severity=0.0, eta_severity=0.0,
                   elliptical_peaks=False),  # 退化为传统 MPB
    ]

    for ci, cfg in enumerate(configs):
        print(f"\nConfig {ci}: dim={cfg.dim}, peaks={cfg.peak_number}, "
              f"irregular={cfg.tau_severity>0}, rotated={cfg.angle_severity>0}")
        b = Benchmark(cfg, seed=int(rng.integers(1 << 30)))

        for env_idx, p in enumerate(b.envs):
            fc = b._fitness(p.positions, p)
            check(f"V1 env{env_idx} 峰中心==峰高", np.allclose(fc, p.heights, atol=1e-8),
                  f"max|f-h|={np.abs(fc - p.heights).max():.2e}")
            check(f"V2 env{env_idx} 最优==max峰高", np.isclose(b.optima[env_idx], p.heights.max()))
            X = rng.uniform(cfg.min_coordinate, cfg.max_coordinate, size=(3000, cfg.dim))
            f = b._fitness(X, p)
            check(f"V3 env{env_idx} f<=最优", np.all(f <= b.optima[env_idx] + 1e-8),
                  f"max f={f.max():.3f}, opt={b.optima[env_idx]:.3f}")
            R = p.rotation[0]
            check(f"V4 env{env_idx} 旋转正交", np.allclose(R @ R.T, np.eye(cfg.dim), atol=1e-9))

    # V5：tau=eta=0 → 变换恒等
    z = rng.uniform(-30, 30, size=(4, 3, 5))
    check("V5 无不规则性时变换恒等",
          np.allclose(transform(z, np.zeros(3), np.zeros((3, 4))), z))

    # V6：误差非负 + V7 可复现
    cfg = configs[1]
    b1 = Benchmark(cfg, seed=99)
    rng2 = np.random.default_rng(0)
    while not b1.done:
        b1.evaluate(rng2.uniform(-50, 50, size=(30, cfg.dim)))
        if b1.changed:
            b1.acknowledge_change()
    check("V6 误差恒非负", np.all(b1.error_history >= 0))

    b2 = Benchmark(cfg, seed=99)
    same = all(np.allclose(e1.positions, e2.positions) and np.allclose(e1.rotation, e2.rotation)
               for e1, e2 in zip(b1.envs, b2.envs))
    check("V7 环境序列可复现", same)

    print("\n" + ("=" * 48))
    print("全部验证通过 ✅" if ok else "存在失败项 ❌")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_checks() else 1)
