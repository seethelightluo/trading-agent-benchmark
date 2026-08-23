"""
Exploration: trend_breadth_20 (regime synchronicity factor)
===========================================================
Idea: In a cross-asset universe, broad-based trends (most assets moving together)
are more reliable than narrow ones. An asset that's moving WITH the prevailing 
market direction should outperform one going against it.

Construction:
1. For each date, compute 20d returns for all assets
2. Determine the median direction (sign of median 20d return across assets)
3. For each asset, compute what fraction of other assets share its direction
4. Multiply by the absolute magnitude of its own return
5. Scale by the absolute median return (regime strength)

This captures: "how aligned is this asset's move with the broad market regime"
- In strong bull: assets with positive returns get high scores (everyone bullish)
- In strong bear: assets with negative returns get positive scores (defensive haven)
- In divergent regimes: no clear signal (score near zero for all)

Expected to complement pure momentum by adding regime-awareness.
"""

import numpy as np
import pandas as pd
import sys, os, json, base64, zlib, io

sys.path.insert(0, 'scripts')

# Override: use current simulation date
CURRENT_DATE = pd.Timestamp("2032-03-04")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS_PER_DATE = 8

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"

def load_closes(end_date=CURRENT_DATE):
    closes, vols = {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        closes[a] = df["close"].astype(float)
        vols[a] = df["volume"].astype(float) if "volume" in df.columns else None
    close = pd.DataFrame(closes)
    vol = pd.DataFrame(vols) if any(v is not None for v in vols.values()) else None
    return close, vol, None, None, None

def load_index(name):
    df = pd.read_csv(f"{INDEX_DIR}/{name}.csv", parse_dates=["date"])
    df = df[df["date"] <= CURRENT_DATE].set_index("date").sort_index()
    return df["close"].astype(float)

def dense_per_asset(close, vol, open_, high, low):
    dense = {}
    for a in ASSETS:
        idx = close[a].dropna().index
        dense[a] = {"close": close[a].reindex(idx)}
    return dense

def factor_panel(fn, close, vol, open_, high, low, macro, **params):
    dense = dense_per_asset(close, vol, open_, high, low)
    out = {}
    for a in ASSETS:
        c = dense[a]["close"]
        try:
            s = fn(c, None, None, None, None, macro, **params)
            out[a] = pd.Series(s.values, index=c.index).reindex(close.index)
        except Exception as e:
            out[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)

def fwd_returns(close, horizon):
    out = {}
    dense = dense_per_asset(close, None, None, None, None)
    for a in ASSETS:
        c = dense[a]["close"]
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)

def ic_series(factor, fwd_ret, min_assets=MIN_ASSETS_PER_DATE):
    dates, ics = [], []
    for dt in factor.index:
        x = factor.loc[dt]
        y = f