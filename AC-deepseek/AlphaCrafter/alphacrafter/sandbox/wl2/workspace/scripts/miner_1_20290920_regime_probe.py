"""miner_1 2029-09-20: regime probe - recent cross-asset moves and data sanity.
Reads data through visible_through (2029-09-19). No persistence, exploration only.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import GRID, ASSETS, load_asset, load_macro

print("grid dates:", len(GRID), "from", GRID[0], "to", GRID[-1])

acct = get_account_dict()
print("watch_list:", acct.get("watch_list"))

rows = []
for s in ASSETS:
    df = load_asset(s, days=400)
    if df is None or len(df) < 60:
        print(s, "NO DATA")
        continue
    c = df["close"].astype(float)
    rets = c.pct_change()
    last = c.iloc[-1]
    r5 = c.iloc[-1] / c.iloc[-6] - 1 if len(c) > 6 else np.nan
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    v60 = rets.tail(60).std() * np.sqrt(252)
    vol_last = df["volume"].iloc[-1] if "volume" in df.columns else np.nan
    vol_med = df["volume"].tail(60).median() if "volume" in df.columns else np.nan
    rows.append((s, r5, r20, r60, v60, vol_last, vol_med, len(df)))
    print(f"{s:10s} r5={r5:+.2%} r20={r20:+.2%} r60={r60:+.2%} annvol60={v60:.1%} "
          f"vol_last={vol_last:.0f} vol_med60={vol_med:.0f} ndays={len(df)}")

for m in ["DXY", "USDJPY", "VIX", "EURUSD", "USDCNY"]:
    ser = load_macro(m)
    if ser is None:
        print(m, "NO DATA")
        continue
    c = ser.astype(float)
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    print(f"{m:8s} last={c.iloc[-1]:.2f} r20={r20:+.2%} r60={r60:+.2%} ndays={len(c)}")

# forward-10d sample availability at the tail
print("\nlast 3 grid rows fwd10 availability check:")
print("grid last dates:", GRID[-3:])
