"""miner_1 (2026-08-11): sweep L - fresh dimension sweep.

Library now covers: momentum (mom_10/120), bb_width, vol_z, rng_pos, skew, kurt,
beta_VIX, days_since_high_60, vix_beta_cond, kaufman_eff. Target NEW orthogonal
dimensions with low library correlation (<0.5):
  - RSI extremes (overbought/oversold mean-reversion)
  - up/down streak length (consistency)
  - Sharpe ratio of short-window momentum (risk-adjusted trend)
  - downside persistence (negative-day clustering)
  - ema cross strength normalized by vol
  - cross-sectional equity-beta dispersion-free hedging proxy

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; prefer max_abs_library_correlation<0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean().replace(0, np.nan)
    rs = up / dn
    return 100.0 - 100.0 / (1.0 + rs)


def streak_len(close, n=14):
    """Weighted winning-streak persistence: run-length weighted share of up days."""
    up = (close.diff() > 0).astype(float)

    def streak_block(x):
        best, cur = 0, 0
        for v in x:
            cur = cur + 1 if v > 0.5 else 0
            best = max(best, cur)
        return best
    return up.rolling(n, min_periods=7).apply(lambda x: streak_block(x.values), raw=False)


def sharpe_mom(close, mom=20, vol=20):
    """Risk-adjusted short momentum: cum return / realized vol."""
    ret = close.pct_change()
    m = close.pct_change(mom)
    v = ret.rolling(vol, min_periods=10).std(ddof=0).replace(0, np.nan)
    return m / v


def down_cluster(close, n=20):
    """Downside persistence: recency-weighted share of negative days."""
    neg = (close.diff() < 0).astype(float)
    w = pd.Series(np.linspace(0.5, 1.0, n), index=range(n))
    return neg.rolling(n, min_periods=10).apply(
        lambda x: float(np.average(np.asarray(x, dtype=float), weights=w.values[:len(x)])), raw=True
    )

def ema_cross_norm(close, fast=10, slow=60):
    """EMA (fast-slow) gap normalized by realized vol (trend strength)."""
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    ret = close.pct_change()
    v = ret.rolling(slow, min_periods=30).std(ddof=0).replace(0, np.nan)
    return (ef / es.replace(0, np.nan) - 1.0) / v.reindex(close.index)


def hi_lo_pos(close, n=30):
    """Close position within trailing high-low range (level persistence)."""
    hi = close.rolling(n, min_periods=15).max()
    lo = close.rolling(n, min_periods=15).min()
    return (close - lo) / (hi - lo).replace(0, np.nan)


candidates = {
    "rsi_7": {a: rsi(closes[a], 7) for a in closes},
    "rsi_14_rev": {a: -(rsi(closes[a], 14) - 50) for a in closes},  # mean-reversion sign
    "streak_len_14": {a: streak_len(closes[a], 14) for a in closes},
    "sharpe_mom_20": {a: sharpe_mom(closes[a], 20, 20) for a in closes},
    "down_cluster_20": {a: down_cluster(closes[a], 20) for a in closes},
    "ema_cross_norm_10x60": {a: ema_cross_norm(closes[a], 10, 60) for a in closes},
    "hi_lo_pos_30": {a: hi_lo_pos(closes[a], 30) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()