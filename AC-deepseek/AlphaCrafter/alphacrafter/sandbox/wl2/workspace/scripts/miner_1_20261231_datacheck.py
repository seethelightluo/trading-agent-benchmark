"""miner_1 2026-12-31 datacheck: verify data through visible_through and print regime snapshot."""
import sys, json
sys.path.insert(0, "scripts")
from miner_1_20261119_lib import series, ASSETS, VISIBLE, GRID, N_GRID
import pandas as pd
import numpy as np

print("visible_through:", VISIBLE, " grid rows:", N_GRID)

for s in ASSETS:
    df = series.get(s)
    if df is None or len(df) < 70:
        continue
    c = df["close"]
    # flat detection
    tail = c.tail(60)
    flat = (tail.diff().abs() < 1e-12).mean()
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    print("%-10s last=%12.4f flat60=%.2f 20d=%8.2f%% 60d=%8.2f%%" % (s, c.iloc[-1], flat, 100 * r20, 100 * r60))

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
