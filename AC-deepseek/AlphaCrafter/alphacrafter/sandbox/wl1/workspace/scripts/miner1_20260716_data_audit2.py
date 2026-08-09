"""Quick audit: data spans, volume availability, fundamental columns (PE/PS/PB/DYR)."""
import os
import pandas as pd

DATA = "../persistent/stock_data"
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

for s in SYMBOLS:
    d = pd.read_csv(f"{DATA}/{s}.csv", parse_dates=["date"])
    d = d.sort_values("date")
    vol = pd.to_numeric(d["volume"], errors="coerce")
    vol_frac = (vol > 0).mean()
    fund = {}
    for col in ["PE", "PS", "PB", "DYR"]:
        if col in d.columns:
            fund[col] = int(d[col].notna().sum())
    print(f"{s:10s} n={len(d):5d} {d['date'].min().date()}..{d['date'].max().date()} "
          f"vol>0: {vol_frac:.3f} fund={fund}")