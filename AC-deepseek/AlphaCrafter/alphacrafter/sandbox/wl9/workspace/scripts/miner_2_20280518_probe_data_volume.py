"""Probe data horizon and volume availability across the 15-asset universe."""
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

print("=== Tradable universe data horizon ===")
for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=2500)
    if df is None or len(df) == 0:
        print(f"{sym}: NO DATA (stock)")
        df = get_index_daily_data(symbol=sym, days=2500)
        if df is not None and len(df) > 0:
            print(f"  -> via index API present, len={len(df)}")
        continue
    vol_ok = df["volume"].notna().sum() if "volume" in df.columns else 0
    print(f"{sym}: len={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()} "
          f"cols={list(df.columns)} vol_nonnull={vol_ok} pct_vol={vol_ok/len(df):.2f}")

print("\n=== Macro signals horizon ===")
import os
for f in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    path = f"../persistent/index_data/{f}.csv"
    if os.path.exists(path):
        m = pd.read_csv(path)
        print(f"{f}: rows={len(m)} cols={list(m.columns)[:6]} first={m.iloc[0,0]} last={m.iloc[-1,0]}")
    else:
        print(f"{f}: MISSING")

# Check date.json horizon (read-only)
with open("../persistent/date.json") as fh:
    print("\ndate.json:", fh.read()[:200])