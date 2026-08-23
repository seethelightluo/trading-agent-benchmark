"""miner_2 (2026-08-19): sweep L - fresh orthogonal dimensions targeting
low library correlations and fresh economic dimensions.

Covered library dims: momentum (short/long), vol level/volume z, range pos,
skew/kurt, macro beta (VIX/DXY), days_since_high, kaufman eff, streak len,
corr-change.

This sweep targets FRESH dims:
  - log-price momentum at 40d skip 10d
  - realized-vol regime change (vol acceleration 5x60)
  - open-to-close within-day efficiency
  - drawdown depth / recovery structure on 60d
  - VIX-z conditioned vol (risk-premium tilt)
  - 30d momentum skip 15 (fresh period)
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; max lib corr < 0.5 for persist.
"""
from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def log_mom(close, n=40, skip=10):
    return np.log(close / close.shift(n + skip))


def vol_accel(close, short=5, long=60):
    r = close.pct_change()
    sv = r.rolling(short, min_periods=3).std(ddof=0)
    lv = r.rolling(long, min_periods=30).std(ddof=0).replace(0, np.nan)
    return sv / lv


def open_close_eff(close, open_, n=20):
    rng = (close / open_.replace(0, np.nan) - 1.0)
    return rng.rolling(n, min_periods=10).mean()


def drawdown_recovery(close, n=60):
    mmax = close.rolling(n, min_periods=30).max()
    return close / mmax.replace(0, np.nan) - 1.0


def cond_vrp(close, n=60):
    vix = macro["VIX"]
    vz = (vix - vix.rolling(60, min_periods=30).mean()) / vix.rolling(60, min_periods=30).std().replace(0, np.nan)
    av = close.pct_change().rolling(n, min_periods=30).std(ddof=0)
    return av.multiply(vz.reindex(av.index).ffill(), axis=0)


def mom_30x15(close):
    return close / close.shift(45) - 1.0


def load_open():
    out = {}
    for a in ASSETS:
        p = f"../persistent/stock_data/{a}.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")["open"].astype(float)
    return out


opens = load_open()

candidates = {
    "log_mom_40x10": {a: log_mom(closes[a], 40, 10) for a in closes},
    "vol_accel_5x60": {a: vol_accel(closes[a], 5, 60) for a in closes},
    "open_close_eff_20d": {a: open_close_eff(closes[a], opens[a], 20) for a in closes},
    "drawdown_rec_60d": {a: drawdown_recovery(closes[a], 60) for a in closes},
    "cond_vrp_60": {a: cond_vrp(closes[a], 60) for a in closes},
    "mom_30x15": {a: mom_30x15(closes[a]) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()