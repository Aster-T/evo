"""投资组合数据：Beasley OR-Library（真实指数）+ yfinance（实时 S&P，best-effort）。

产出统一为 `(mu, Sigma, names)`：
  mu    —— 各资产期望收益 (N,)
  Sigma —— 收益协方差矩阵 (N, N)，对称半正定
  names —— 资产标签（用于绘图/报告）

Beasley 数据：Chang, Meade, Beasley, Sharaiha (2000) 的 port1–port5，分别对应
Hang Seng(31) / DAX 100(85) / FTSE 100(89) / S&P 100(98) / Nikkei 225(225)。
文件格式：
  第 1 行：资产数 N
  接下来 N 行：每资产的「平均收益 标准差」
  其后若干行：「i j ρ_ij」给出相关系数（上三角，1-based）
协方差 Σ = D ρ D，其中 D = diag(std)。

yfinance：抓取一批大盘股日收益，算 (mu, Sigma) 后把收盘价缓存到 data/sp500_prices.csv；
抓取失败（如 Yahoo 限流 429）则返回 None，由实验层优雅跳过该实例。

用法（下载/缓存）：
  conda run -n evo python -m src.data.portfolio_data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
BEASLEY_DIR = DATA_DIR / "beasley"
BEASLEY_BASE = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

BEASLEY_MARKETS = {
    "port1.txt": "Hang Seng (31)",
    "port2.txt": "DAX 100 (85)",
    "port3.txt": "FTSE 100 (89)",
    "port4.txt": "S&P 100 (98)",
    "port5.txt": "Nikkei 225 (225)",
}

# 一篮子大盘股（用于 yfinance 自建组合；约 50 只，行业分散）
SP50_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "ABBV",
    "MRK", "COST", "AVGO", "MCD", "DIS", "CSCO", "ADBE", "NKE", "INTC", "VZ",
    "CMCSA", "PFE", "T", "ORCL", "CRM", "ABT", "TXN", "QCOM", "DHR", "UNH",
    "LIN", "PM", "HON", "UNP", "LOW", "IBM", "GS", "CAT", "AMGN", "BA",
]


# ---------------------------------------------------------------- Beasley
def _download_beasley(fname: str) -> Path:
    """下载单个 Beasley 文件到 data/beasley/（已存在则跳过）。"""
    BEASLEY_DIR.mkdir(parents=True, exist_ok=True)
    dst = BEASLEY_DIR / fname
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    import requests
    url = BEASLEY_BASE + fname
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dst.write_text(resp.text, encoding="utf-8")
    return dst


def load_beasley(fname: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """解析 Beasley portN.txt → (mu, Sigma, names)。"""
    path = _download_beasley(fname)
    tokens = path.read_text(encoding="utf-8").split()
    it = iter(tokens)
    n = int(next(it))
    mu = np.empty(n)
    std = np.empty(n)
    for i in range(n):
        mu[i] = float(next(it))
        std[i] = float(next(it))
    corr = np.eye(n)
    # 剩余三元组 i j rho（1-based，含对角）
    rest = list(it)
    for k in range(0, len(rest), 3):
        i = int(float(rest[k])) - 1
        j = int(float(rest[k + 1])) - 1
        rho = float(rest[k + 2])
        corr[i, j] = rho
        corr[j, i] = rho
    D = np.diag(std)
    Sigma = D @ corr @ D
    names = [f"A{i+1}" for i in range(n)]
    return mu, Sigma, names


# ---------------------------------------------------------------- yfinance
def fetch_sp500(tickers: list[str] | None = None, period: str = "3y",
                cache: bool = True):
    """抓取一批股票日收益 → (mu, Sigma, names)；失败返回 None（best-effort）。

    收益用日对数收益的均值/协方差；价格缓存到 data/sp500_prices.csv 以便离线复现。
    """
    tickers = tickers or SP50_TICKERS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "sp500_prices.csv"

    prices: pd.DataFrame | None = None
    if cache and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        try:
            import yfinance as yf
            raw = yf.download(tickers, period=period, interval="1d",
                              auto_adjust=True, progress=False, threads=True)
            # 多列时取 Close 层；单列时直接用
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            prices = prices.dropna(axis=1, how="any")
            if prices.shape[1] < 5:
                return None
            if cache:
                prices.to_csv(cache_path)
        except Exception as e:  # 限流 / 网络失败 → 跳过
            print(f"[fetch_sp500] 抓取失败，跳过 yfinance 实例：{e}")
            return None

    rets = np.log(prices / prices.shift(1)).dropna()
    mu = rets.mean().to_numpy()
    Sigma = rets.cov().to_numpy()
    names = list(prices.columns)
    return mu, Sigma, names


def main() -> None:
    print("下载/缓存 Beasley OR-Library …")
    for fname, label in BEASLEY_MARKETS.items():
        mu, Sigma, names = load_beasley(fname)
        print(f"  {fname:10s} {label:18s} N={len(mu):3d}  "
              f"mu∈[{mu.min():.4f},{mu.max():.4f}]  Σ shape={Sigma.shape}")
    print("尝试 yfinance 自建 S&P 组合（best-effort）…")
    out = fetch_sp500()
    if out is None:
        print("  → 未取得行情（限流/网络），实验将仅用 Beasley。")
    else:
        mu, Sigma, names = out
        print(f"  → 成功：{len(names)} 只股票，已缓存 data/sp500_prices.csv")


if __name__ == "__main__":
    main()
