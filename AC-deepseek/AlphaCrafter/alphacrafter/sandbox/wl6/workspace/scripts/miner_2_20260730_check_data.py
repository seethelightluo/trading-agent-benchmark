import sys, math
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_validation_lib import TRADABLE, load_panel, load_macro

panel = load_panel()
print("panel shape:", panel.shape)
print("date range:", panel.index.min().date(), "->", panel.index.max().date())
print("assets:", panel.columns.tolist())

# per-asset data spans and coverage
for c in panel.columns:
    s = panel[c].dropna()
    print(f"  {c:<10} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")

# volume availability via simulator API
from alphacrafter.sim.utils import get_stock_daily_data
vol_ok = {}
for sym in TRADABLE:
    try:
        df = get_stock_daily_data(symbol=sym, days=4000)
        if df is not None and "volume" in df:
            v = df["volume"].dropna()
            vol_ok[sym] = (len(v) > 100 and float(v.iloc[-50:].mean()) > 0)
        else:
            vol_ok[sym] = False
    except Exception:
        vol_ok[sym] = False
print("volume available:", vol_ok)

# macro spans
for m in ["DXY", "VIX", "USDJPY", "USDCNY", "EURUSD"]:
    s = load_macro(m)
    print(f"  macro {m:<8} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")

# OHLC availability (high/low for range factors)
import warnings
warnings.filterwarnings("ignore")
for sym in ["000300.SH", "BTC", "XAU", "US10Y", "SPX"]:
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is not None:
        print(f"  {sym:<10} cols={list(df.columns)} rows={len(df)}")
