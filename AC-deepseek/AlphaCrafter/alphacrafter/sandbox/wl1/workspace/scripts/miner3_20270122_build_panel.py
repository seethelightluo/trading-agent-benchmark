"""miner_3: rebuild panel cache from CSV files up to current sim date (2027-01-22).
Only uses data <= 2027-01-22 (no future leakage). Saves scripts/panel_cache.pkl.
"""
import pandas as pd
import numpy as np
import glob

CUR = pd.Timestamp("2027-01-22")
SYMS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
        "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

def load(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= CUR].set_index("date").sort_index()
    return df

# price panel (15 tradable)
close = {}
opn = {}
high = {}
low = {}
vol = {}
for s in SYMS:
    df = load(f"../persistent/stock_data/{s}.csv")
    close[s] = df["close"]
    opn[s] = df["open"]
    high[s] = df["high"]
    low[s] = df["low"]
    vol[s] = df["volume"]
C = pd.DataFrame(close).sort_index()
O = pd.DataFrame(opn).sort_index()
H = pd.DataFrame(high).sort_index()
L = pd.DataFrame(low).sort_index()
V = pd.DataFrame(vol).sort_index()
# common index (drop pure NaN rows)
idx = C.dropna(how="all").index
C, O, H, L, V = C.loc[idx], O.loc[idx], H.loc[idx], L.loc[idx], V.loc[idx]
ret = C.pct_change()

# macro panel
mac = {}
for m in MACRO:
    df = load(f"../persistent/index_data/{m}.csv")
    mac[m] = df["close"]
M = pd.DataFrame(mac).sort_index()
M = M.reindex(C.index)

panel = {"close": C, "open": O, "high": H, "low": L, "vol": V, "ret": ret, "macro": M}
panel["close"].to_pickle("scripts/panel_cache.pkl")
import pickle
with open("scripts/panel_cache.pkl", "wb") as fh:
    pickle.dump(panel, fh, protocol=4)

print("panel rebuilt:", C.shape, C.index.min().date(), "->", C.index.max().date())
print("macro last valid:", M.apply(lambda s: s.dropna().index.max()).to_dict())
