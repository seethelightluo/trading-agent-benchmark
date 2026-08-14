"""miner_2 2034-05-25 probe: data availability + current tape snapshot through VISIBLE 2034-05-24."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner_3_20260813_lib import ASSETS, GRID, N_GRID, VISIBLE, load_asset, load_macro

print("VISIBLE:", VISIBLE, "| grid rows:", N_GRID, "| assets:", len(ASSETS))
print("ASSETS:", ASSETS)

# per-asset data availability
avail = {}
for s in ASSETS:
    df = load_asset(s, days=4000)
    if df is None:
        avail[s] = 0
        continue
    avail[s] = len(df)
print("rows per asset:", avail)

# recent returns snapshot (as of last visible date)
snap = {}
last_date = None
for s in ASSETS:
    df = load_asset(s, days=4000)
    if df is None or len(df) < 200:
        continue
    c = df["close"].astype(float)
    last_date = df.index[-1]
    snap[s] = {
        "close": float(c.iloc[-1]),
        "r5": float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 else np.nan,
        "r20": float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else np.nan,
        "r60": float(c.iloc[-1] / c.iloc[-61] - 1) if len(c) > 61 else np.nan,
        "r180": float(c.iloc[-1] / c.iloc[-181] - 1) if len(c) > 181 else np.nan,
        "r252": float(c.iloc[-1] / c.iloc[-253] - 1) if len(c) > 253 else np.nan,
    }
print("\nsnapshot date:", last_date)
t = pd.DataFrame(snap).T
print(t.round(4).to_string())

# macro recent levels
for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    s = load_macro(m)
    if s is None:
        print("macro missing:", m); continue
    s = s.dropna()
    if len(s) < 30:
        print("macro short:", m, len(s)); continue
    last = s.index[-1]
    r20 = s.iloc[-1] / s.iloc[-21] - 1 if len(s) > 21 else np.nan
    r60 = s.iloc[-1] / s.iloc[-61] - 1 if len(s) > 61 else np.nan
    print(f"macro {m}: last={s.iloc[-1]:.2f} (date {last}) r20={r20*100:.2f}% r60={r60*100:.2f}%")
