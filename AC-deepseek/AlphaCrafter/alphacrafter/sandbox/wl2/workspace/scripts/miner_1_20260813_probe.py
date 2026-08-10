"""Probe data availability as of current simulation date 2026-08-13."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("../persistent/stock_data")
INDEX_DIR = Path("../persistent/index_data")

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

print("=== stock_data ===")
for f in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(f, parse_dates=["date"])
    print(f"{f.name:12s} rows={len(df):5d} last={df['date'].max().date()} first={df['date'].min().date()}")

print("=== index_data ===")
for f in sorted(INDEX_DIR.glob("*.csv")):
    df = pd.read_csv(f, parse_dates=["date"])
    print(f"{f.name:12s} rows={len(df):5d} last={df['date'].max().date()} first={df['date'].min().date()}")

# check columns
df = pd.read_csv(DATA_DIR / "SPX.csv", parse_dates=["date"])
print("SPX columns:", list(df.columns))
print(df.tail(3).to_string())
