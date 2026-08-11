"""miner2 2027-02-19: build fresh daily panel from persistent CSVs (tradable + macro)."""
import pandas as pd
import numpy as np
import pickle

SYMS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
        "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load(path, cols=None):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if cols is not None:
        df = df[cols]
    return df


close, opn, high, low, vol = {}, {}, {}, {}, {}
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
idx = C.dropna(how="all").index
C, O, H, L, V = C.loc[idx], O.loc[idx], H.loc[idx], L.loc[idx], V.loc[idx]
ret = C.pct_change()

mac = {}
for m in MACRO:
    df = load(f"../persistent/index_data/{m}.csv")
    mac[m] = df["close"]
M = pd.DataFrame(mac).sort_index().reindex(C.index)

panel = {"close": C, "open": O, "high": H, "low": L, "vol": V, "ret": ret, "macro": M}
with open("scripts/miner2_panel.pkl", "wb") as fh:
    pickle.dump(panel, fh, protocol=4)

print("panel:", C.shape, C.index.min().date(), "->", C.index.max().date())
print("cols:", list(C.columns))
print("macro last:", {k: str(v.dropna().index.max().date()) for k, v in M.items()})
print("n_valid_per_asset:", C.notna().sum().to_dict())
