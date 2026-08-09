"""Miner3 quick data check: date ranges and volume availability per symbol."""
import os
import pandas as pd

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA_DIR = "../persistent/stock_data"
CUT = pd.Timestamp("2026-07-15")

print(f"{'sym':10s} {'rows':>6s} {'start':>12s} {'end':>12s} {'vol_nz%':>8s} {'vol>0':>7s}")
for s in SYMBOLS:
    d = pd.read_csv(os.path.join(DATA_DIR, f"{s}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= CUT].sort_values("date")
    vol = pd.to_numeric(d["volume"], errors="coerce")
    nz = (vol > 0).mean()
    print(f"{s:10s} {len(d):6d} {str(d['date'].iloc[0].date()):>12s} {str(d['date'].iloc[-1].date()):>12s} "
          f"{100*nz:7.1f}% {int((vol>0).sum()):7d}")

# macro files
IDX_DIR = "../persistent/index_data"
print("\nmacro files:")
for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    d = pd.read_csv(os.path.join(IDX_DIR, f"{m}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= CUT]
    print(f"  {m:8s} rows={len(d):6d} {str(d['date'].iloc[0].date())}..{str(d['date'].iloc[-1].date())}")
