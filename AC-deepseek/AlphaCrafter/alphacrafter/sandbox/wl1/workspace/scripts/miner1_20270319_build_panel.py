"""miner_1 2027-03-19: rebuild research panel from CSV files up to 2027-03-18
(previous completed trading day; no future leakage). Saves scripts/panel_cache.pkl.
"""
import pandas as pd
import numpy as np
import pickle

CUR = pd.Timestamp("2027-03-18")
SYMS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
        "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

def load(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= CUR].set_index("date").sort_index()
    return df

close, opn, high, low, vol = {}, {}, {}, {}, {}
for s in SYMS:
    df = load(f"../persistent/stock_data/{s}.csv")
    close[s] = df["close"]; opn[s] = df["open"]; high[s] = df["high"]
    low[s] = df["low"]; vol[s] = df["volume"]
C = pd.DataFrame(close).sort_index()
O = pd.DataFrame(opn).sort_index()
H = pd.DataFrame(high).sort_index()
L = pd.DataFrame(low).sort_index()
V = pd.DataFrame(vol).sort_index()
idx = C.dropna(how="all").index
C, O, H, L, V = C.loc[idx], O.loc[idx], H.loc[idx], L.loc[idx], V.loc[idx]
ret = C.pct_change()

mac = {}
for m in MACRO:
    df = load(f"../persistent/index_data/{m}.csv")
    mac[m] = df["close"]
M = pd.DataFrame(mac).sort_index().reindex(C.index)

panel = {"close": C, "open": O, "high": H, "low": L, "vol": V, "ret": ret, "macro": M}
with open("scripts/panel_cache.pkl", "wb") as fh:
    pickle.dump(panel, fh, protocol=4)

print("panel rebuilt:", C.shape, C.index.min().date(), "->", C.index.max().date())
print("macro last valid:", {k: str(v.dropna().index.max().date()) for k, v in M.items()})
wd = C.index.dayofweek < 5
print("weekday rows:", int(wd.sum()))
valid_cnt = C[wd].notna().sum(axis=1)
print("weekday dates with >=8 valid:", int((valid_cnt >= 8).sum()))
print("per-asset last date:", {k: str(v.dropna().index.max().date()) for k, v in C.items()})
