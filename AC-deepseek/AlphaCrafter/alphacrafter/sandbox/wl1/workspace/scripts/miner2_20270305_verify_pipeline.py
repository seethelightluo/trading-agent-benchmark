"""miner2 2027-03-05: verify pipeline vs prior metrics; compare CSV panel vs API calendar."""
import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

# --- API data check ---
acct = get_account_dict()
print("watch_list from account:", acct.get("watch_list"))
for s in ["SPX", "BTC", "XAU"]:
    df = get_stock_daily_data(symbol=s, days=3000)
    print(s, "api rows:", 0 if df is None else len(df),
          "first:", df["date"].iloc[0] if df is not None else None,
          "last:", df["date"].iloc[-1] if df is not None else None)

# --- CSV panel coverage ---
p = pickle.load(open("scripts/miner2_panel.pkl", "rb"))
C = p["close"]
print("\nCSV panel rows per asset (sample):")
for s in ["SPX", "BTC", "XAU", "WTI"]:
    print(s, C[s].notna().sum(), "first:", C[s].dropna().index.min().date(), "last:", C[s].dropna().index.max().date())

# --- reproduce nclv_1d IC over the ORIGINAL persisted window 2021-01-01..2026-07-15 ---
H, L = p["high"], p["low"]
fdf = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
for a in C.columns:
    fdf[a] = -(C[a] - L[a]) / (H[a] - L[a])
w = fdf.loc["2021-01-01":"2026-07-15"]
fr = C.shift(-1) / C - 1.0
frw = fr.loc[w.index]
ics = []
for i in range(len(w)):
    fv = w.iloc[i].values; rv = frw.iloc[i].values
    m = np.isfinite(fv) & np.isfinite(rv)
    if m.sum() >= 8:
        ics.append(spearmanr(fv[m], rv[m]).correlation)
ics = np.array(ics)
print("\n[CSV panel] nclv_1d IC1 2021-01-01..2026-07-15: n=%d ic=%.4f icir=%.4f (persisted: ic=0.0649 icir=0.183 n=1171)" % (
    len(ics), np.nanmean(ics), np.nanmean(ics) / np.nanstd(ics)))
