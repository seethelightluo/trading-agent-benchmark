"""miner_2 (2026-08-20): sweep N - fresh dimensions not covered by library.

Covered library dims: momentum short/long, vol z, bb width, kurt, skew, rng_pos,
macro beta (VIX/DXY/CNY), days_since_high, kaufman eff, streak len, dxy_corr_change.

This sweep targets FRESH dims:
  - return autocorrelation lag-1 (over 20/60d): daily mean-reversion/trend memory
  - momentum acceleration: short vs medium momentum spread (regime change)
  - volatility-scaled momentum (return/vol) with a 20d horizon
  - downside deviation ratio (semi-deviation share) - lower partial moment
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; persist needs max lib corr < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes

closes = load_closes()


def autocorr_lag1(close, n=20):
    r = close.pct_change()
    r0 = r.rolling(n, min_periods=n // 2).mean()
    r1 = r.shift(1)
    cov = ((r - r0) * (r1 - r0)).rolling(n, min_periods=n // 2).mean()
    var = ((r - r0) ** 2).rolling(n, min_periods=n // 2).mean().replace(0, np.nan)
    return cov / var


def mom_accel(close, short=10, med=40):
    """Short momentum minus medium momentum: recent acceleration / regime change."""
    ms = close / close.shift(short) - 1.0
    mm = close / close.shift(med) - 1.0
    return ms - mm


def vol_mom(close, n=20, vol_window=20):
    """Volatility-scaled momentum: return over n days per unit realized vol."""
    r = close.pct_change()
    vol = r.rolling(vol_window, min_periods=vol_window // 2).std(ddof=0).replace(0, np.nan)
    mom = close / close.shift(n) - 1.0
    return mom / vol


def downside_dev_ratio(close, n=20):
    """Ratio of downside semi-deviation to total realized vol: lower-partial-moment tilt."""
    r = close.pct_change()
    mn = r.rolling(n, min_periods=n // 2).mean()
    downside = (r - mn).clip(upper=0.0)
    sd = ((r - mn) ** 2).rolling(n, min_periods=n // 2).mean() ** 0.5
    dsd = (downside ** 2).rolling(n, min_periods=n // 2).mean() ** 0.5
    return dsd / sd.replace(0, np.nan)


candidates = {
    "ac1_20d": {a: autocorr_lag1(closes[a], 20) for a in closes},
    "ac1_60d": {a: autocorr_lag1(closes[a], 60) for a in closes},
    "ac1_120d": {a: autocorr_lag1(closes[a], 120) for a in closes},
    "mom_accel_10x40": {a: mom_accel(closes[a], 10, 40) for a in closes},
    "mom_accel_20x60": {a: mom_accel(closes[a], 20, 60) for a in closes},
    "vol_mom_20": {a: vol_mom(closes[a], 20, 20) for a in closes},
    "vol_mom_40": {a: vol_mom(closes[a], 40, 20) for a in closes},
    "downside_dev_ratio_20": {a: downside_dev_ratio(closes[a], 20) for a in closes},
    "downside_dev_ratio_60": {a: downside_dev_ratio(closes[a], 60) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()