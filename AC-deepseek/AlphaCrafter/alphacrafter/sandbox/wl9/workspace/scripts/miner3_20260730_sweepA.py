"""Exploration sweep A (miner_3, 2026-07-30): trend/price-structure families.

Candidates tested at admission horizon 10 against the shared gates:
  abs(IC) >= 0.0070 and abs(ICIR) >= 0.0840.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20260730_harness import load_closes, evaluate

closes = load_closes()


def rng_pos(close, high, low, n=20):
    hi = high.rolling(n).max()
    lo = low.rolling(n).min()
    rng = (hi - lo).replace(0, np.nan)
    return (close - lo) / rng


def dist_high(close, high, n=252):
    return close / high.rolling(n).max() - 1.0


def mom_vol_ratio(close, n=20):
    mom = close / close.shift(n) - 1.0
    vol = close.pct_change().rolling(n).std()
    return mom / vol.replace(0, np.nan)


def skew_20(close, n=20):
    r = close.pct_change()
    m = r.rolling(n).mean()
    sd = r.rolling(n).std(ddof=0).replace(0, np.nan)
    return ((r - m) ** 3).rolling(n).mean() / (sd ** 3)


def pos_frac(close, n=60, skip=0):
    r = close.pct_change()
    up = (r > 0).astype(float)
    return up.shift(skip).rolling(n).mean()


def bbz(close, n=20):
    sma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0).replace(0, np.nan)
    return (close - sma) / sd


# ---- per-asset high/low series ----
highs = {a: pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).set_index("date")["high"].astype(float)
         for a in closes}
lows = {a: pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).set_index("date")["low"].astype(float)
        for a in closes}

candidates = {
    "rng_pos_20d": {a: rng_pos(closes[a], highs[a], lows[a], 20) for a in closes},
    "dist_high_252": {a: dist_high(closes[a], highs[a], 252) for a in closes},
    "mom_vol_ratio_20": {a: mom_vol_ratio(closes[a], 20) for a in closes},
    "skew_20d": {a: skew_20(closes[a], 20) for a in closes},
    "pos_frac_60d": {a: pos_frac(closes[a], 60, 0) for a in closes},
    "bbz_20d": {a: bbz(closes[a], 20) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", e)
    print()
