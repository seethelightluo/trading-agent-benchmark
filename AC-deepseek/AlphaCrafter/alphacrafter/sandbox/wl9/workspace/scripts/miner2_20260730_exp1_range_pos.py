"""Exploration 1: Range-position trend factor (stochastic %K style).

Idea: where the close sits within the recent high-low range measures trend
strength/persistence. Sustained uptrends keep close near range highs; this may
predict continuation across the cross-asset universe. Uses high/low/close
(12 of 15 assets have valid OHLC; CN10Y/SOX/US10Y lack high/low).

Construction: (close - min(low, w)) / (max(high, w) - min(low, w)), clipped to [0,1].
Variants: w in {10, 20, 40, 60}, plus smoothed versions.
"""
import sys, json
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import load_panel, factor_panel, fwd_ret_panel, validate, load_close
import pandas as pd

P = load_panel()

# Load OHLC for range computation
def load_ohlc(symbol):
    import os
    df = pd.read_csv(os.path.join("../persistent/stock_data", f"{symbol}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp("2026-07-29")]
    df = df.set_index("date")[["high", "low", "close"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()

OHLC = {a: load_ohlc(a) for a in P.columns}

def range_pos(hi, lo, cl, w):
    rng = hi.rolling(w).max() - lo.rolling(w).min()
    pos = (cl - lo.rolling(w).min()) / rng
    return pos.clip(0, 1)

results = []
for w in [10, 20, 40, 60]:
    fvals = pd.DataFrame({a: range_pos(OHLC[a]["high"], OHLC[a]["low"], OHLC[a]["close"], w) for a in P.columns}).sort_index()
    fvals = fvals[~fvals.index.duplicated(keep="last")]
    fwd10 = fwd_ret_panel(P, 10)
    res = validate(fvals, fwd10, label=f"range_pos_{w}", expected_dir=1)
    res["coverage_assets"] = int(fvals.notna().sum(axis=0).gt(0).sum())
    results.append(res)
    print(json.dumps(res))
