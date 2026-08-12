"""miner_1 2026-11-19 datacheck: verify data through visible_through and print regime snapshot."""
import json, sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20261119_lib import series, ASSETS, GRID, N_GRID, VISIBLE

print("VISIBLE:", VISIBLE, "N_GRID:", N_GRID)

# asset coverage in last 120 grid rows
last = GRID[-121:]
for s in ASSETS:
    df = series.get(s)
    if df is None:
        print(s, "MISSING")
        continue
    sub = df[df.index.isin(last)]
    nonflat = (sub["close"].diff() != 0).sum()
    print("%-10s rows=%4d last=%s last_close=%12.4f nonflat_last120=%d" % (
        s, len(df), df.index[-1], df['close'].iloc[-1], nonflat))

# 20/60d returns snapshot
print("\n--- 20d/60d returns (own calendar) ---")
for s in ASSETS:
    df = series.get(s)
    if df is None or len(df) < 70:
        continue
    c = df["close"]
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    print("%-10s 20d=%8.2f%% 60d=%8.2f%%" % (s, 100 * r20, 100 * r60))

# observation-only macro signals
for sym in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    try:
        df = pd.read_csv("../persistent/index_data/%s.csv" % sym)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.set_index("date")
        col = [c for c in df.columns if c.lower() in ("close", "value", "price")][0]
        c = df[col].astype(float)
        c = c[c.index <= VISIBLE]
        if len(c) < 70:
            print(sym, "insufficient", len(c)); continue
        r20 = c.iloc[-1] / c.iloc[-21] - 1
        r60 = c.iloc[-1] / c.iloc[-61] - 1
        print("%-7s last=%10.3f 20d=%7.2f%% 60d=%7.2f%% n=%d" % (sym, c.iloc[-1], 100 * r20, 100 * r60, len(c)))
    except Exception as e:
        print(sym, "ERR", e)
