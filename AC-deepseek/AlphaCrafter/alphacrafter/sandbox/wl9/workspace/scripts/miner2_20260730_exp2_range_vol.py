"""Exploration 2: Range-vol ratio (Parkinson volatility regime / compression).

Idea: the ratio of short-horizon intraday range volatility to long-horizon
range volatility measures whether an asset's realized vol regime is compressing
or expanding. Range-based vol is more efficient than close-to-close vol and
uses different info than the existing vol_of_vol factor.

Construction: pv(w) = mean(log(high/low)^2) over w days (Parkinson estimator);
factor = pv(short)/pv(long)  (range compression) and  pv(short)-pv(long) variant.
Also test a simple (mean high-low range)/close ratio (ATR-style) compression.
"""
import sys, json, os
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import load_panel, factor_panel, fwd_ret_panel, validate
import pandas as pd, numpy as np

P = load_panel()

def load_ohlc(symbol):
    df = pd.read_csv(os.path.join("../persistent/stock_data", f"{symbol}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp("2026-07-29")]
    df = df.set_index("date")[["high", "low", "close"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()

OHLC = {a: load_ohlc(a) for a in P.columns}

def parkinson(hi, lo, w):
    r = (np.log(hi / lo)) ** 2
    return np.sqrt(r.rolling(w).mean())

def atr_ratio(hi, lo, cl, w):
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(w).mean() / cl

results = []
for (short, long) in [(5, 20), (5, 60), (10, 40), (10, 60), (20, 60)]:
    fvals = pd.DataFrame({a: (parkinson(OHLC[a]["high"], OHLC[a]["low"], short) /
                              parkinson(OHLC[a]["high"], OHLC[a]["low"], long))
                          for a in P.columns}).sort_index()
    fvals = fvals[~fvals.index.duplicated(keep="last")]
    res = validate(fvals, fwd_ret_panel(P, 10), label=f"park_ratio_{short}x{long}", expected_dir=-1)
    results.append(res)
    print(json.dumps(res))

# ATR-ratio compression (expected: compressed vol -> future expansion? test both signs via abs)
for (short, long) in [(10, 60), (20, 60)]:
    fvals = pd.DataFrame({a: (atr_ratio(OHLC[a]["high"], OHLC[a]["low"], OHLC[a]["close"], short) /
                              atr_ratio(OHLC[a]["high"], OHLC[a]["low"], OHLC[a]["close"], long))
                          for a in P.columns}).sort_index()
    fvals = fvals[~fvals.index.duplicated(keep="last")]
    res = validate(fvals, fwd_ret_panel(P, 10), label=f"atr_ratio_{short}x{long}", expected_dir=-1)
    results.append(res)
    print(json.dumps(res))
