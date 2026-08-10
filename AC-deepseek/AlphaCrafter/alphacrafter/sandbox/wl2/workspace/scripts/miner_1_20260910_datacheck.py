"""miner_1 2026-09-10: data availability check for 15-asset universe."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

print("=== per-asset data check (visible through 2026-09-09) ===")
for s in ASSETS:
    df = get_stock_daily_data(s, days=2100)
    if df is None:
        print(f"{s}: NO DATA")
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    last = df["date"].iloc[-1]
    vol_ok = pd.to_numeric(df["volume"], errors="coerce").notna().sum()
    vol_nz = (pd.to_numeric(df["volume"], errors="coerce") > 0).sum()
    print(f"{s}: rows={len(df)} first={df['date'].iloc[0]} last={last} "
          f"vol_nonnull={vol_ok} vol_nz={vol_nz}")
