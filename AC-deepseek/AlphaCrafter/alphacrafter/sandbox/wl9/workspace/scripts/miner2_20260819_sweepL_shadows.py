"""miner_2 (2026-08-19): sweep L - candlestick SHADOW factors (fresh dimension).

Library covers momentum, bandwidth, vol, range-close-position (rng_pos), skew,
kurtosis, macro-beta, efficiency, streak, kaufman. Not yet covered: the
asymmetric location of the wick/shadows relative to the open-close body.

Candidates (family: intraday-range shadow asymmetry / body placement):
  - upper_shadow_20d: mean over 20d of (high - max(open,close)) / (high-low)
  - lower_shadow_20d: mean over 20d of (min(open,close) - low) / (high-low)
  - shadow_asym_20d: (upper - lower) shadow ratio (net upper-wick pressure)
  - body_pos_20d: mean of (close-open)/(high-low) scaled body direction vs range

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; persistence also needs
max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes

closes = load_closes()

def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()


def upper_shadow(df, n=20):
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    op = df["open"].astype(float)
    cl = df["close"].astype(float)
    rng = (hi - lo).replace(0, np.nan)
    body_hi = np.maximum(op, cl)
    sh = (hi - body_hi) / rng
    return sh.rolling(n, min_periods=10).mean()


def lower_shadow(df, n=20):
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    op = df["open"].astype(float)
    cl = df["close"].astype(float)
    rng = (hi - lo).replace(0, np.nan)
    body_lo = np.minimum(op, cl)
    sh = (body_lo - lo) / rng
    return sh.rolling(n, min_periods=10).mean()


def shadow_asym(df, n=20):
    """Net upper-wick pressure: mean upper shadow minus mean lower shadow share."""
    return upper_shadow(df, n) - lower_shadow(df, n)


def body_pos(df, n=20):
    """Mean signed body-to-range: (close-open)/(high-low) - close-position tilt."""
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    op = df["open"].astype(float)
    cl = df["close"].astype(float)
    rng = (hi - lo).replace(0, np.nan)
    body = (cl - op) / rng
    return body.rolling(n, min_periods=10).mean()


candidates = {
    "upper_shadow_20d": {a: upper_shadow(ohlc[a], 20) for a in closes},
    "lower_shadow_20d": {a: lower_shadow(ohlc[a], 20) for a in closes},
    "shadow_asym_20d": {a: shadow_asym(ohlc[a], 20) for a in closes},
    "body_pos_20d": {a: body_pos(ohlc[a], 20) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()