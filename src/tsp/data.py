"""TSPLIB 真实城市坐标加载（真实数据）。

数据来源：TSPLIB（Reinelt 1991），真实地理/电路坐标，进化算法求解 TSP 的标准基准。
Heidelberg 官网当前不稳定，改用 GitHub 镜像 mastqe/tsplib 下载 `.tsp` 文件并缓存到
data/tsplib/。仅支持 EUC_2D（二维欧氏距离，TSPLIB 约定 d_ij = round(sqrt(dx²+dy²))）。

已知最优巡回长度（best-known/optimal，用于计算 gap%）来自 TSPLIB 官方解。

用法（下载/缓存）：
  conda run -n evo python -m src.tsp.data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tsplib"
MIRROR = "https://raw.githubusercontent.com/mastqe/tsplib/master/{inst}.tsp"

# 已知最优巡回长度（TSPLIB 官方最优解）
OPTIMA: dict[str, int] = {
    "berlin52": 7542,
    "eil51": 426,
    "st70": 675,
    "eil76": 538,
    "kroA100": 21282,
    "ch130": 6110,
    "ch150": 6528,
    "a280": 2579,
}


def _download(inst: str) -> Path:
    """下载 <inst>.tsp 到 data/tsplib/（已存在则跳过）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / f"{inst}.tsp"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    import requests
    resp = requests.get(MIRROR.format(inst=inst), timeout=30)
    resp.raise_for_status()
    dst.write_text(resp.text, encoding="utf-8")
    return dst


def load_coords(inst: str) -> np.ndarray:
    """解析 TSPLIB EUC_2D 实例 → 坐标数组 (n, 2)。"""
    path = _download(inst)
    coords = []
    in_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("NODE_COORD_SECTION"):
            in_section = True
            continue
        if s == "EOF":
            break
        if in_section:
            parts = s.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    if not coords:
        raise ValueError(f"{inst}: 未解析到坐标（可能非 EUC_2D 格式）")
    return np.array(coords, dtype=float)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """EUC_2D 距离矩阵（四舍五入取整，TSPLIB 约定）。"""
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return np.rint(d)


def main() -> None:
    print("下载/缓存 TSPLIB 实例 …")
    for inst in ("berlin52", "eil51", "kroA100", "ch150", "a280"):
        coords = load_coords(inst)
        D = distance_matrix(coords)
        print(f"  {inst:10s} n={len(coords):4d}  最优={OPTIMA.get(inst,'?')}  "
              f"D对称={np.allclose(D, D.T)}")


if __name__ == "__main__":
    main()
