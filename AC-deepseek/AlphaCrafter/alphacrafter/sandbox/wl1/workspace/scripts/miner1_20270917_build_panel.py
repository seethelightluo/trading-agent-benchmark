"""miner_1: rebuild panel cache from CSV files up to 2027-09-16 (visible_through).
Current date 2027-09-17; daily OHLCV visible only through previous completed trading day.
No future leakage. Saves scripts/panel_cache.pkl (overwrites prior miner panel).
"""
import pandas as pd
import numpy as np
import pickle

CUR = pd.Timestamp("2027-09-16")
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
    mac[m] = load(f"../persistent/index_data/{m}.csv")["close"]
M = pd.DataFrame(mac).sort_index().reindex(C.index)

panel = {"close": C, "open": O, "high": H, "low": L, "vol": V, "ret": ret, "macro": M}
with open("scripts/panel_cache.pkl", "wb") as fh:
    pickle.dump(panel, fh, protocol=4)

print("panel rebuilt:", C.shape, C.index.min().date(), "->", C.index.max().date())
print("assets:", list(C.columns))
print("macro last valid:", {m: str(M[m].dropna().index.max().date()) for m in MACRO})
print("per-asset rows min/max:", int(C.notna().sum().min()), int(C.notna().sum().max()))
print("total dates:", len(C))
