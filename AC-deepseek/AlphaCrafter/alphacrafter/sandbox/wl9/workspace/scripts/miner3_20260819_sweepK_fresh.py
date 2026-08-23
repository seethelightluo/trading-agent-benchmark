"""miner_3 (2026-08-19): Sweep K - fresh time-series dimensions.

Library now: mom_10/120, bb_width, vol_z, rng_pos, skew, beta_VIX, days_since_high_60,
vix_beta_cond, kaufman_eff_20d, kurt_20d, streak_len_14, dxy_corr_change_20_60.
We target dimensions NOT yet covered in the effective library:
  - RSI(14) and RSI(7) -> short-term overbought/oversold mean reversion
  - return autocorrelation (persistence to complement mom)
  - MA crossover spread (MA10/MA60, trend alignment sign)
  - drawdown depth 60d / 20d (pullback / max drawdown)
  - intraday range autocorrelation / overnight-gap mean reversion
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; persistence needs max lib corr < 0.5.
"""
from __future__ import annotations
import sys, io, base64, zlib, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes

closes = load_closes()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean().replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + up / dn)


def autocorr(close, n=20, lag=1):
    r = close.pct_change()
    def ac(x):
        x = x.dropna()
        if len(x) < 10:
            return np.nan
        return x.autocorr(lag=lag)
    return r.rolling(n, min_periods=10).apply(lambda x: x.autocorr(lag=lag), raw=False)


def ma_cross(close, s=10, l=60):
    m_s = close.rolling(s).mean()
    m_l = close.rolling(l).mean()
    return (m_s / m_l) - 1.0


def max_drawdown(close, n=60):
    rollmax = close.rolling(n, min_periods=n).max()
    return (close / rollmax) - 1.0


def down_up_vol_ratio(close, n=20, minp=4):
    r = close.pct_change()
    neg = r.where(r < 0).rolling(n, min_periods=minp).std(ddof=0)
    pos = r.where(r > 0).rolling(n, min_periods=minp).std(ddof=0)
    return pos / neg.replace(0, np.nan)


# candidate: overnight-gap persistence: today's open gap vs yesterday close, mean over n
def gap_persist(close, n=5):
    return close.pct_change(n)


candidates = {
    "rsi_14": {a: rsi(closes[a], 14) for a in closes},
    "rsi_7": {a: rsi(closes[a], 7) for a in closes},
    "autocorr_20_lag1": {a: autocorr(closes[a], 20, 1) for a in closes},
    "autocorr_60_lag1": {a: autocorr(closes[a], 60, 1) for a in closes},
    "ma_cross_10x60": {a: ma_cross(closes[a], 10, 60) for a in closes},
    "max_drawdown_60d": {a: max_drawdown(closes[a], 60) for a in closes},
    "max_drawdown_20d": {a: max_drawdown(closes[a], 20) for a in closes},
    "down_up_vol_20d": {a: down_up_vol_ratio(closes[a], 20) for a in closes},
}

print("assets:", len(closes))
for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()