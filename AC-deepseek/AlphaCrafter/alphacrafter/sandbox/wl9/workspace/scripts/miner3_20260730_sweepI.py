"""Exploration sweep I (miner_3, 2026-07-30): fresh distinct factor families.

Targets novel, interpretable factor ideas not heavily represented in the
current library. Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on the
15-asset universe; persistence requires max_abs_library_correlation < 0.5.

Candidates:
1) sr_zscore_10x60 : short-term overreaction vs long-term level
   (close/ewm60 - 1) standardized / realized vol -> mean-reversion signal
2) updown_vol_ratio_60 : upside volatility / downside volatility asymmetry
3) drawdown_252 : 1 - close/rolling_max(close,252)  (distance from 52w high)
4) semi_vol_60 : downside semi-deviation contribution to total vol (risk measure)
5) eff_ratio_60s : Kaufman efficiency ratio 60d
6) rv5_rv20 : ratio of 5d realized vol to 20d (short-term vol breakout)
7) csi_rel_120 : relative strength vs CSI300 (local benchmark tilt)
8) gold_beta_60 : rolling beta of each asset to XAU (safe-haven correlation)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes  # noqa: E402

closes = load_closes()
print("assets:", len(closes))


def sr_zscore(close, s=10, l=60):
    fast = close / close.ewm(l, adjust=False).mean() - 1.0
    rv = close.pct_change().rolling(s, min_periods=5).std(ddof=0).replace(0, np.nan)
    return fast / rv


def updown_vol_ratio(close, n=60, minp=20):
    r = close.pct_change()
    pos = r.where(r > 0, 0.0)
    neg = r.where(r < 0, 0.0)
    up = np.sqrt((pos ** 2).rolling(n, min_periods=minp).mean())
    dn = np.sqrt((neg ** 2).rolling(n, min_periods=minp).mean())
    return (up / dn.replace(0, np.nan)).replace(np.inf, np.nan)


def drawdown_252(close, n=252):
    return 1.0 - close / close.rolling(n, min_periods=n).max()


def semi_vol_60(close, n=60, minp=20):
    r = close.pct_change()
    neg = r.where(r < 0, 0.0)
    ds = np.sqrt((neg ** 2).rolling(n, min_periods=minp).mean())
    tot = r.rolling(n, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return ds / tot


def eff_ratio_60s(close, n=60):
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n).sum()
    return num / den.replace(0, np.nan)


def rv5_rv20(close):
    r = close.pct_change()
    v5 = r.rolling(5, min_periods=3).std(ddof=0)
    v20 = r.rolling(20, min_periods=10).std(ddof=0)
    return v5 / v20.replace(0, np.nan)


def csi_rel_120(close, csi, n=120):
    a = close / close.shift(n)
    b = csi / csi.shift(n)
    return a - b


def gold_beta_60(close, gold, n=60):
    ra = close.pct_change()
    rb = gold.pct_change()
    cov = ra.rolling(n).cov(rb)
    var = rb.rolling(n).var(ddof=0).replace(0, np.nan)
    return cov / var


csi = closes["000300.SH"]
gold = closes["XAU"]

candidates = {
    "sr_zscore_10x60": {a: sr_zscore(closes[a]) for a in closes},
    "updown_vol_60": {a: updown_vol_ratio(closes[a], 60) for a in closes},
    "drawdown_252": {a: drawdown_252(closes[a]) for a in closes},
    "semi_vol_60": {a: semi_vol_60(closes[a]) for a in closes},
    "eff_ratio_60s": {a: eff_ratio_60s(closes[a]) for a in closes},
    "rv5_rv20": {a: rv5_rv20(closes[a]) for a in closes},
    "csi_rel_120": {a: csi_rel_120(closes[a], csi, 120) for a in closes},
    "gold_beta_60": {a: gold_beta_60(closes[a], gold, 60) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()