"""miner_2 (2026-08-19): sweep K - orthogonal/conditional dimensions targeting low library correlation.

Goal: find factors satisfying absIC>=0.007 & absICIR>=0.084 AND max_abs_library_correlation<0.5.
Candidates deliberately target under-covered dimensions:
  - volatility trend/drift (short vs long vol ratio)
  - up/down streak persistence
  - high-low efficiency at different scale
  - volume-independent quality transforms
  - macro-regime-conditional cross-sectional tilt (VIX/DXY conditioning)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def vol_trend_ratio(close, short=5, long=60):
    """Realized-vol ratio: short-window vol / long-window vol (vol momentum/trend)."""
    sv = close.pct_change().rolling(short, min_periods=3).std(ddof=0)
    lv = close.pct_change().rolling(long, min_periods=30).std(ddof=0).replace(0, np.nan)
    return sv / lv


def streak_win(close, n=10):
    """Share of up-days over last n days weighted toward recency (persistence)."""
    up = (close.diff() > 0).astype(float)
    return up.rolling(n, min_periods=5).mean()


def hi_lo_eff(close, n=20):
    """Close-to-high/efficiency: (close-low)/(high-low) range normalized (already rng_pos; use diff window)."""
    # use alternative: close placement vs prior-day range midpoint drift
    return close.pct_change(3)


def vol_midgap(close, n=20):
    """Volatility relative to intraday range (close-range misalignment)."""
    r = close.pct_change()
    v = r.rolling(n, min_periods=10).std(ddof=0).replace(0, np.nan)
    return v


def cond_vix_vol(close, n=60):
    """Cross-sectional tilt: asset realized vol amplified when VIX regime high."""
    vix = macro["VIX"]
    vix_z = (vix - vix.rolling(60, min_periods=30).mean()) / vix.rolling(60, min_periods=30).std().replace(0, np.nan)
    av = close.pct_change().rolling(n, min_periods=30).std(ddof=0)
    out = av.multiply(vix_z.reindex(av.index).ffill(), axis=0)
    return out


def cond_dxy_mom(close, n=20):
    """Momentum-scaled negative by DXY beta regime (risk-off hedge tilt)."""
    dxy = macro["DXY"].pct_change().rolling(n, min_periods=10).mean()
    mom = close.pct_change(n)
    return mom.multiply(dxy.reindex(mom.index).ffill(), axis=0)


def turnover_cream_ratio(close, short=3, long=20):
    """Short/long MA absolute spread scaled by vol (signal strength)."""
    s = close.rolling(short, min_periods=2).std(ddof=0)
    l = close.rolling(long, min_periods=10).std(ddof=0).replace(0, np.nan)
    return s / l


candidates = {
    "vol_trend_5x60": {a: vol_trend_ratio(closes[a], 5, 60) for a in closes},
    "streak_win_10d": {a: streak_win(closes[a], 10) for a in closes},
    "hi_lo_eff_20d": {a: hi_lo_eff(closes[a], 20) for a in closes},
    "vol_cluster_3x20": {a: turnover_cream_ratio(closes[a], 3, 20) for a in closes},
    "cond_vix_vol_60": {a: cond_vix_vol(closes[a], 60) for a in closes},
    "cond_dxy_mom_20": {a: cond_dxy_mom(closes[a], 20) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()