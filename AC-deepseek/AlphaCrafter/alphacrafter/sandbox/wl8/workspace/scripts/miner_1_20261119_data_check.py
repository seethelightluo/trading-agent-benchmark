"""miner_1 2026-11-19: data freshness & availability check (direct CSV reads)."""
import pandas as pd
import numpy as np

WATCHLIST = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
STOCK_DIR = "../persistent/stock_data/"
INDEX_DIR = "../persistent/index_data/"

print("=== TRADABLE ASSETS ===")
panels = {}
for s in WATCHLIST:
    try:
        df = pd.read_csv(f"{STOCK_DIR}/{s}.csv")
    except Exception as e:
        print(f"{s}: READ FAIL {e}")
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    last = df["date"].max().date()
    tail5 = df["close"].tail(6)
    tail_chg = tail5.pct_change().abs().sum()
    vol_ok = ("volume" in df.columns) and df["volume"].tail(5).astype(float).abs().sum() > 0
    print(f"{s}: rows={len(df)} last={last} tail5_abs_chg={tail_chg:.4f} has_vol_tail={vol_ok} cols={list(df.columns)}")
    panels[s] = df

print("\n=== MACRO FEEDS ===")
for m in MACRO:
    df = pd.read_csv(f"{INDEX_DIR}/{m}.csv")
    df["date"] = pd.to_datetime(df["date"])
    print(f"{m}: rows={len(df)} last={df['date'].max().date()} cols={list(df.columns)}")

# Build a common close panel to see what per-date cross-section looks like
print("\n=== COMMON DATE GRID ===")
dates = None
for s, df in panels.items():
    dts = set(pd.to_datetime(df["date"]).dt.normalize())
    dates = dts if dates is None else dates & dts
dates = sorted(dates)
print(f"common trading dates: {len(dates)} from {dates[0].date()} to {dates[-1].date()}")

close = pd.DataFrame({s: df.set_index("date")["close"] for s, df in panels.items()})
close = close.reindex(dates)
valid_dates = close.notna().sum(axis=1)
print(f"dates with >=8 assets: {(valid_dates >= 8).sum()}")
print("last 5 valid-dates:", close.index[-5:].strftime('%Y-%m-%d').tolist())
print("asset-days total:", int(close.notna().sum().sum()), "of", int(close.shape[0]*close.shape[1]))