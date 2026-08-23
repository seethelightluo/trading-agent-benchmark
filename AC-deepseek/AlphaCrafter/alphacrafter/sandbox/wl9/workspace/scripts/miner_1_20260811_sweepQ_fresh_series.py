"""miner_1 (2026-08-cycle): sweep Q - FRESH factor series for the 15-asset cross-asset universe.

Universe note explicitly confirms 15 intentionally tradable instruments and
min_valid>=8 is sufficient for a daily IC observation. This harness uses the
shared miner3_20260730_harness validation pipeline.

Fresh dimensions vs current library (price-return mom/vol/skew/kurt/streak,
DXY/VIX corr-beta, volume z-score):
  1. close_loc_20: 20d avg of (close-low)/(high-low) -- intraday close bias/pressure
  2. ohlc_gap_mom: momentum of the open-to-close (daily body) component
  3. vol_adj_mom_break: 20d momentum divided by 20d realized vol (risk-adjusted trend)
  4. updown_vol_ratio_20: rolling up-vol / down-vol asymmetry
  5. close_vs_vwap20: close / 20d average-close as a drift-normalized trend
  6. vol_convexity: (high-low)/close 20d mean scaled by close/MA20 (vol-trend interaction)

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; prefer max lib corr<0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import (
    ASSETS, evaluate, load_closes, load_macro, STOCK_DIR, VISIBLE_END,
)

def load_ohlc():
    """Load full OHLCV (not just close) per asset, capped to VISIBLE_END."""
    out = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END].set_index("date")
        out[a] = df
    return out

closes = load_closes()
ohlc = load_ohlc()

def body_frame(ohlc):
    """open-to-close percent series per asset."""
    return {a: (df["close"] / df["open"] - 1.0) for a, df in ohlc.items()}

def range_frame(ohlc):
    return {a: (df["high"] - df["low"]) / df["close"] for a, df in ohlc.items()}

def close_loc(ohlc, n=20):
    """20d mean of intraday close position (close-low)/(high-low)."""
    out = {}
    for a, df in ohlc.items():
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        loc = (df["close"] - df["low"]) / rng
        out[a] = loc.rolling(n, min_periods=10).mean()
    return out

def gap_mom(ohlc, n=20):
    """Momentum of the daily body (open-to-close) component, 20d sum."""
    out = {}
    for a, df in ohlc.items():
        body = (df["close"] / df["open"] - 1.0)
        out[a] = body.rolling(n, min_periods=10).sum()
    return out

def vol_adj_mom(closes, n=20):
    """20d momentum / 20d realized vol."""
    out = {}
    for a, s in closes.items():
        r = s.pct_change()
        mom = s / s.shift(n) - 1.0
        vol = r.rolling(n, min_periods=10).std(ddof=0).replace(0, np.nan)
        out[a] = mom / vol
    return out

def updown_vol_ratio(closes, n=20):
    """upside-vol / downside-vol over n, log."""

    out = {}
    for a, s in closes.items():
        r = s.pct_change()
        up = r.clip(lower=0)
        dn = r.clip(upper=0)
        uv = up.rolling(n, min_periods=10).std(ddof=0).replace(0, np.nan)
        dv = dn.rolling(n, min_periods=10).std(ddof=0).replace(0, np.nan).abs().replace(0, np.nan)
        out[a] = np.log(uv / dv)
    return out

def close_vs_ma20(closes, n=20):
    """close / rolling mean close - 1 (drift-normalized trend, not raw mom)."""
    out = {}
    for a, s in closes.items():
        ma = s.rolling(n, min_periods=10).mean()
        out[a] = s / ma - 1.0
    return out

def vol_convexity(closes, ohlc, n=20):
    """(high-low)/close 20d mean * sign(close/MA20 - 1) interaction."""
    out = {}
    for a, s in closes.items():
        df = ohlc[a]
        rng = (df["high"] - df["low"]) / df["close"]
        rng20 = rng.rolling(n, min_periods=10).mean()
        ma = s.rolling(n, min_periods=10).mean()
        out[a] = rng20 * np.sign(s / ma - 1.0)
    return out

print("close count:", len(closes), "ohlc count:", len(ohlc))
print("date range:", min(closes[a].index.min() for a in closes),
      "..", max(closes[a].index.max() for a in closes))

cand = {
    "close_loc_20": close_loc(ohlc),
    "gap_mom_20": gap_mom(ohlc),
    "vol_adj_mom_20": vol_adj_mom(closes),
    "updown_vol_ratio_20": updown_vol_ratio(closes),
    "close_vs_ma20": close_vs_ma20(closes),
    "vol_convexity_20": vol_convexity(closes, ohlc),
}

for label, vals in cand.items():
    try:
        evaluate(closes, vals, label, horizon=10, verbose=True)
    except Exception as e:
        print(f"=== {label} === ERROR: {e}")