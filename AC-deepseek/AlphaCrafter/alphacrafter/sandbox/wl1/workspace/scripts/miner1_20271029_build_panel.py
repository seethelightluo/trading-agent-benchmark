"""miner_1: rebuild panel cache through 2027-10-28 (visible through prev trading day; current date 2027-10-29)."""
import pandas as pd
import numpy as np
import pickle

CUR = pd.Timestamp("2027-10-28")
SYMS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
        "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

def load(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= CUR].set_index("date").sort_index()
    return df

close, opn, high, low, vol = {}, {}, {}, {}, {}
for s in SYMS:
    d = load(f"../persistent/stock_data/{s}.csv")
    close[s] = d["close"]; opn[s] = d["open"]; high[s] = d["high"]
    low[s] = d["low"]; vol[s] = d["volume"]
C = pd.DataFrame(close).sort_index(); O = pd.DataFrame(opn).sort_index()
H = pd.DataFrame(high).sort_index(); L = pd.DataFrame(low).sort_index()
V = pd.DataFrame(vol).sort_index()
idx = C.dropna(how="all").index
C, O, H, L, V = C.loc[idx], O.loc[idx], H.loc[idx], L.loc[idx], V.loc[idx]
ret = C.pct_change()
mac = {m: load(f"../persistent/index_data/{m}.csv")["close"] for m in MACRO}
M = pd.DataFrame(mac).sort_index().reindex(C.index)

with open("scripts/panel_cache.pkl", "wb") as fh:
    pickle.dump({"close": C, "open": O, "high": H, "low": L, "vol": V, "ret": ret, "macro": M}, fh, protocol=4)
print("panel cached:", C.shape, C.index.min().date(), "->", C.index.max().date())
print("macro last:", {m: str(M[m].dropna().index.max().date()) for m in MACRO})
