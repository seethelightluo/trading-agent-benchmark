"""miner_1: rebuild panel cache through visible 2028-04-27 (current date 2028-04-28)."""
import os
import pickle
import numpy as np
import pandas as pd

CUR = pd.Timestamp("2028-04-27")
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"


def load(dir_, syms):
    out = {}
    for s in syms:
        d = pd.read_csv(os.path.join(dir_, f"{s}.csv"))
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] <= CUR].set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        out[s] = d
    return out


C = pd.DataFrame({s: load(DATA_DIR, [s])[s]["close"] for s in SYMBOLS}).dropna(how="all")
O = pd.DataFrame({s: load(DATA_DIR, [s])[s]["open"] for s in SYMBOLS}).reindex(C.index)
H = pd.DataFrame({s: load(DATA_DIR, [s])[s]["high"] for s in SYMBOLS}).reindex(C.index)
L = pd.DataFrame({s: load(DATA_DIR, [s])[s]["low"] for s in SYMBOLS}).reindex(C.index)
V = pd.DataFrame({s: load(DATA_DIR, [s])[s]["volume"] for s in SYMBOLS}).reindex(C.index)
ret = C.pct_change()
M = pd.DataFrame({s: load(IDX_DIR, [s])[s]["close"] for s in MACRO}).reindex(C.index)

panel = {"close": C, "open": O, "high": H, "low": L, "vol": V, "ret": ret, "macro": M}
with open("scripts/panel_cache.pkl", "wb") as fh:
    pickle.dump(panel, fh, protocol=4)
print("panel rebuilt:", C.shape, C.index.min().date(), "->", C.index.max().date())
print("macro cols:", list(M.columns), "last macro date:", M.index.max().date())
