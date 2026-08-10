"""Miner3 quick data check: date ranges, volume availability, and NaN structure."""
import os
import numpy as np
import pandas as pd

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA_DIR = "../persistent/stock_data"
CUT = pd.Timestamp("2026-07-15")

closes = {}
for s in SYMBOLS:
    d = pd.read_csv(os.path.join(DATA_DIR, f"{s}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= CUT].sort_values("date").set_index("date")
    closes[s] = d

idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
print(f"common index: {len(idx)} dates {idx.min().date()}..{idx.max().date()}")

print(f"\n{'sym':10s} {'rows':>6s} {'vol>0%':>7s} {'vol_nan%':>8s} {'ohlc_nan%':>8s}")
for s in SYMBOLS:
    d = closes[s]
    vol = pd.to_numeric(d["volume"], errors="coerce")
    nz = 100 * (vol > 0).mean()
    vnan = 100 * vol.isna().mean()
    ohlc = d[["open", "high", "low", "close"]].isna().mean().mean() * 100
    print(f"{s:10s} {len(d):6d} {nz:7.1f} {vnan:8.1f} {ohlc:8.1f}")
