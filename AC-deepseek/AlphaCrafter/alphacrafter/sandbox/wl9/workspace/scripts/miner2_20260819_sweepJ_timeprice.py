"""miner_2 (2026-08-19): sweep J - time-series price dynamics dimensions NOT yet covered.

Library now: mom_10/120, bb_width, vol_z, rng_pos, skew, beta_VIX, days_since_high_60,
vix_beta_cond, kaufman_eff_20d, kurt_20d.
We target fresh dimensions:
  - RSI (overbought/oversold mean-reversion)
  - return autocorrelation (persistence/reversal)
  - MA crossover spread (MA10/MA60, trend alignment)
  - max drawdown / drawdown depth
  - downside/upside vol ratio (asymmetry)
  - gap/overnight return persistence
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; persistence needs max lib corr < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()


def rsi(close, n=14):
    """Wilder RSI over n days."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean().replace(0, np.nan)
    rs = up / dn
    return 100.0 - 100.0 / (1.0 + rs)


def autocorr(close, n=20, lag=1):
    """Rolling lag-1 autocorrelation of daily returns over n days."""
    r = close.pct_change()
    m = r.rolling(n, min_periods=10).mean()
    def ac(x):
        x = x.dropna()
        if len(x) < 10:
            return np.nan
        return x.autocorr(lag=lag)
    return r.rolling(n, min_periods=10).apply(lambda x: x.autocorr(lag=lag), raw=False)


def ma_cross(close, short=10, long=60):
    """Short/long MA ratio minus 1 (trend alignment)."""
    ma_s = close.rolling(short, min_periods=5).mean()
    ma_l = close.rolling(long, min_periods=30).mean()
    return ma_s / ma_l.replace(0, np.nan) - 1.0


def max_drawdown(close, n=20):
    """Depth of trailing max drawdown: close/max(close,n) - 1 (negative = drawdown)."""
    roll_max = close.rolling(n, min_periods=10).max()
    return close / roll_max.replace(0, np.nan) - 1.0


def down_up_vol_ratio(close, n=20):
    """Downside-vol / upside-vol ratio (return asymmetry / fear)."""
    r = close.pct_change()
    down = r.clip(upper=0)
    up = r.clip(lower=0)
    dv = down.rolling(n, min_periods=10).std(ddof=0).replace(0, np.nan)
    uv = up.rolling(n, min_periods=10).std(ddof=0).replace(0, np.nan)
    return dv / uv


def gap_ret(close, n=5):
    """Overnight-gap-type move persistence: today vs n days ago open-close proxy.
    Here use close/prev_close - spread over n days as gap momentum proxy."""
    return close.pct_change(n)


candidates = {
    "rsi_14": {a: rsi(closes[a], 14) for a in closes},
    "autocorr_20_lag1": {a: autocorr(closes[a], 20, 1) for a in closes},
    "ma_cross_10x60": {a: ma_cross(closes[a], 10, 60) for a in closes},
    "max_drawdown_20d": {a: max_drawdown(closes[a], 20) for a in closes},
    "down_up_vol_20d": {a: down_up_vol_ratio(closes[a], 20) for a in closes},
    "gap_mom_5d": {a: gap_ret(closes[a], 5) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()