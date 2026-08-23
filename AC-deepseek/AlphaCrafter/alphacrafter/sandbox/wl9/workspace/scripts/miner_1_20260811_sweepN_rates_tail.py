"""miner_1 (2026-08-11): sweep N - rates/carry, tail risk, trend consistency.

Fresh dimensions vs library:
  - carry_spread: US10Y - CN10Y yield differential (cross-asset rates carry proxy)
  - tail_ratio_60: high-quantile / low-quantile daily return ratio (fat-tail asymmetry)
  - trend_consistency_60: share of days close>SMA (trend persistence, distinct from kaufman)
  - rsi_slope_20: slope/change of RSI over 20d (momentum of oscillator)
  - skew_ratio_60: mean/vol of 60d returns (semi-static quality, different from skew_20d)
  - cross_mom_dispersion: within-day cross-sectional mom dispersion not used
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; prefer lib corr<0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def carry_spread():
    us = macro["US10Y"]
    cn = macro["CN10Y"]
    r = (us - cn).rename("carry")
    out = {a: r for a in closes}
    return out


def tail_ratio(close, n=60, ql=0.1, qh=0.9):
    r = close.pct_change()
    lo = r.rolling(n, min_periods=40).quantile(ql)
    hi = r.rolling(n, min_periods=40).quantile(qh)
    return (hi - lo) / (hi + lo).abs().replace(0, np.nan)


def trend_consistency(close, n=60):
    ma = close.rolling(n, min_periods=30).mean()
    return (close > ma).astype(float).rolling(n, min_periods=30).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean().replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + (up / dn))


def rsi_slope(close, n=20, rsi_n=14):
    return rsi(close, rsi_n).diff(n)


def skew_ratio(close, n=60):
    r = close.pct_change()
    m = r.rolling(n, min_periods=40).mean()
    v = r.rolling(n, min_periods=40).std(ddof=0).replace(0, np.nan)
    return m / v


def down_vol_share(close, n=60):
    r = close.pct_change()
    down = r.clip(upper=0).rolling(n, min_periods=40).std(ddof=0)
    up = r.clip(lower=0).rolling(n, min_periods=40).std(ddof=0).replace(0, np.nan)
    return down / (down + up)


def drawdown_recovery(close, n=60):
    """Time-to-recover from prior peak scaled - 1/(1+days since lowest)."""
    roll_max = close.rolling(n, min_periods=30).max()
    mdd = close / roll_max.replace(0, np.nan) - 1.0
    return mdd


candidates = {
    "carry_yield_spread": carry_spread(),
    "tail_ratio_60": {a: tail_ratio(closes[a], 60) for a in closes},
    "trend_consistency_60": {a: trend_consistency(closes[a], 60) for a in closes},
    "rsi_slope_20": {a: rsi_slope(closes[a], 20) for a in closes},
    "skew_ratio_60": {a: skew_ratio(closes[a], 60) for a in closes},
    "down_vol_share_60": {a: down_vol_share(closes[a], 60) for a in closes},
    "drawdown_60d": {a: drawdown_recovery(closes[a], 60) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()