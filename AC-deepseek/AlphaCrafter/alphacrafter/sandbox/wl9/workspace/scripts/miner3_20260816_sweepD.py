"""Exploration sweep D (miner_3, 2026-08-16): genuinely distinct factor families.

Motivation: hi_lo_range_20d is now redundant with the persisted bb_width_20d
(pairwise signal rho 0.93 -> would be evicted). This sweep tests NEW families:
1) return serial autocorrelation (lag-1), 2) downside/upside semi-vol asymmetry,
3) candle shadow ratio (wick structure), 4) short/medium realized-vol slope,
5) overnight (gap) mean return, 6) rolling beta to SPX (market beta),
7) longer-window return skewness (skew_60 vs library skew_20), 8) up-day ratio
(drift freq), 9) volume participation z-score (10 assets w/ real volume),
10) Amihud illiquidity z-score (10 assets), 11) MACD histogram.

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on the 15-asset universe,
plus max_abs_library_correlation < 0.5 for persistence.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes  # noqa: E402

closes = load_closes()
VOL_ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "NDX", "BTC", "ETH"]


def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()


def autocorr_60(close, n=60):
    r = close.pct_change()
    return r.rolling(n).corr(r.shift(1))


def downside_ratio_20(close, n=20):
    r = close.pct_change()
    neg = r.where(r < 0).rolling(n).std(ddof=0)
    pos = r.where(r > 0).rolling(n).std(ddof=0)
    return neg / pos.replace(0, np.nan)


def shadow_ratio_20(high, low, close, n=20):
    up = (high - close).rolling(n).sum()
    dn = (close - low).rolling(n).sum()
    return up / dn.replace(0, np.nan)


def vol_ratio_5x20(close):
    r = close.pct_change()
    rv5 = r.rolling(5).std(ddof=0)
    rv20 = r.rolling(20).std(ddof=0)
    return rv5 / rv20.replace(0, np.nan)


def overnight_mean_20(open_, close, n=20):
    gap = open_ / close.shift(1) - 1.0
    return gap.rolling(n).mean()


def beta_spx_60(close, spx, n=60):
    ra = close.pct_change()
    rb = spx.pct_change()
    cov = ra.rolling(n).cov(rb)
    var = rb.rolling(n).var(ddof=0).replace(0, np.nan)
    return cov / var


def skew_60(close, n=60):
    r = close.pct_change()
    m = r.rolling(n).mean()
    sd = r.rolling(n).std(ddof=0).replace(0, np.nan)
    m3 = ((r - m) ** 3).rolling(n).mean()
    return m3 / (sd ** 3)


def up_day_ratio_60(close, n=60):
    return (close.diff() > 0).astype(float).rolling(n).mean()


def vol_z_20(volume, n=20):
    if volume.nunique() <= 1:
        return pd.Series(np.nan, index=volume.index)
    mu = volume.rolling(n).mean()
    sd = volume.rolling(n).std(ddof=0).replace(0, np.nan)
    return (volume - mu) / sd


def amihud_z_20(close, volume, n=20):
    if volume.nunique() <= 1:
        return pd.Series(np.nan, index=volume.index)
    ill = (close.pct_change().abs() / volume.replace(0, np.nan))
    mu = ill.rolling(n).mean()
    sd = ill.rolling(n).std(ddof=0).replace(0, np.nan)
    return (ill - mu) / sd


def macd_12_26(close):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    return (e12 - e26) / close.replace(0, np.nan)


spx = closes["SPX"]

vol_assets = {a: ohlc[a]["volume"].astype(float) for a in VOL_ASSETS}

candidates = {
    "autocorr_60d": {a: autocorr_60(closes[a], 60) for a in closes},
    "downside_ratio_20d": {a: downside_ratio_20(closes[a], 20) for a in closes},
    "shadow_ratio_20d": {a: shadow_ratio_20(ohlc[a]["high"].astype(float), ohlc[a]["low"].astype(float), closes[a], 20) for a in closes},
    "vol_ratio_5x20": {a: vol_ratio_5x20(closes[a]) for a in closes},
    "overnight_mean_20d": {a: overnight_mean_20(ohlc[a]["open"].astype(float), closes[a], 20) for a in closes},
    "beta_spx_60": {a: beta_spx_60(closes[a], spx, 60) for a in closes},
    "skew_60d": {a: skew_60(closes[a], 60) for a in closes},
    "up_day_ratio_60d": {a: up_day_ratio_60(closes[a], 60) for a in closes},
    "vol_z_20d": {a: vol_z_20(vol_assets[a], 20) if a in vol_assets else pd.Series(np.nan, index=closes[a].index) for a in closes},
    "amihud_z_20d": {a: amihud_z_20(closes[a], vol_assets[a], 20) if a in vol_assets else pd.Series(np.nan, index=closes[a].index) for a in closes},
    "macd_12_26": {a: macd_12_26(closes[a]) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()