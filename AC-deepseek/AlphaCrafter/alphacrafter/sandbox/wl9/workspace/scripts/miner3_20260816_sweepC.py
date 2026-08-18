"""Exploration sweep C (miner_3, 2026-08-16): trend-efficiency, risk-adjusted
quality, drawdown, and oscillation families.

Distinct from existing library (mom skip-5, vol-of-vol, vix-beta, rng_pos,
skew, dside_ratio, macro betas). Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import evaluate, load_closes  # noqa: E402

closes = load_closes()


def kaufman_eff(close, n=60):
    """Kaufman efficiency ratio: |close - close.shift(n)| / sum(|diff|, n).
    Path-efficiency of the trend: high = smooth trending, low = choppy."""
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n).sum()
    return num / den.replace(0, np.nan)


def sharpe_60(close, n=60):
    """Rolling Sharpe: mean daily return / std daily return (annualization-free)."""
    r = close.pct_change()
    sd = r.rolling(n).std(ddof=0).replace(0, np.nan)
    return r.rolling(n).mean() / sd


def max_dd_60(close, n=60):
    """Trailing max drawdown (negative): close / rolling_max(close, n) - 1."""
    return close / close.rolling(n).max() - 1.0


def rsi_14(close, n=14):
    """Classic RSI: 100 - 100/(1 + avg_gain/avg_loss)."""
    r = close.diff()
    gain = r.where(r > 0, 0.0)
    loss = (-r).where(r < 0, 0.0)
    ag = gain.rolling(n, min_periods=max(6, n // 2)).mean()
    al = loss.rolling(n, min_periods=max(6, n // 2)).mean()
    rs = ag / al.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def bb_width_20(close, n=20):
    """Bollinger bandwidth: 4*std(close,n)/SMA(close,n) — volatility level."""
    sma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return 4.0 * sd / sma.replace(0, np.nan)


def kurt_60(close, n=60):
    """Excess kurtosis of daily returns."""
    r = close.pct_change()
    m = r.rolling(n).mean()
    sd = r.rolling(n).std(ddof=0).replace(0, np.nan)
    m4 = ((r - m) ** 4).rolling(n).mean()
    return m4 / (sd ** 4) - 3.0


def hi_lo_range_20(close, high, low, n=20):
    """(rolling_max(high,n)-rolling_min(low,n))/close — range width normalized."""
    hi = high.rolling(n).max()
    lo = low.rolling(n).min()
    return (hi - lo) / close.replace(0, np.nan)


def streak_5(close, n=5):
    """Consecutive up-day streak: count of last n days with r>0 (run momentum)."""
    r = (close.diff() > 0).astype(float)
    out = []
    streak = 0.0
    for v in r:
        if v == 1:
            streak += 1.0
        else:
            streak = 0.0
        out.append(streak)
    return pd.Series(out, index=r.index).clip(upper=n)


highs = {a: pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).set_index("date")["high"].astype(float)
         for a in closes}
lows = {a: pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).set_index("date")["low"].astype(float)
        for a in closes}

candidates = {
    "kaufman_eff_60d": {a: kaufman_eff(closes[a], 60) for a in closes},
    "sharpe_60d": {a: sharpe_60(closes[a], 60) for a in closes},
    "max_dd_60d": {a: max_dd_60(closes[a], 60) for a in closes},
    "rsi_14d": {a: rsi_14(closes[a], 14) for a in closes},
    "bb_width_20d": {a: bb_width_20(closes[a], 20) for a in closes},
    "kurt_60d": {a: kurt_60(closes[a], 60) for a in closes},
    "hi_lo_range_20d": {a: hi_lo_range_20(closes[a], highs[a], lows[a], 20) for a in closes},
    "streak_5d": {a: streak_5(closes[a], 5) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()