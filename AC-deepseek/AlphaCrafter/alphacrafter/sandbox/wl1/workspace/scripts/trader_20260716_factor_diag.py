"""Trader: compute the 14 library factors on the panel and inspect cross-correlations."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acc = get_account_dict()
assets = acc["watch_list"]
N = 140
frames = {}
for a in assets:
    df = get_stock_daily_data(a, days=N)
    if df is not None and len(df) >= 80:
        frames[a] = df.sort_values("date").reset_index(drop=True)

# align on common dates
dates = None
for a, df in frames.items():
    s = set(df["date"].astype(str))
    dates = s if dates is None else dates & s
dates = sorted(dates)
print("common dates:", len(dates))

px = {a: df.set_index(df["date"].astype(str)).reindex(dates) for a, df in frames.items()}
def col(a, c):
    return px[a][c].astype(float).values

close = pd.DataFrame({a: col(a, "close") for a in frames}, index=dates)
open_ = pd.DataFrame({a: col(a, "open") for a in frames}, index=dates)
high = pd.DataFrame({a: col(a, "high") for a in frames}, index=dates)
low = pd.DataFrame({a: col(a, "low") for a in frames}, index=dates)

lr = np.log(close / close.shift(1))

def cs_z(x):
    return (x - x.mean(axis=1)) / x.std(axis=1)

factors = {}
factors["cz_rev1"] = -cs_z(lr)
factors["rev1_pk"] = -lr / np.sqrt((np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2)))
eff = np.abs(close / close.shift(20) - 1) / lr.abs().rolling(20).sum()
factors["rev1_x_inveff"] = lr * (1 - eff)
factors["id_rev_1d"] = -(close / open_ - 1)
factors["nbody_1d"] = -(close - open_) / (high - low)
for nd in (1, 2, 3, 5):
    factors[f"nclv_{nd}d"] = -(close - low.rolling(nd).min()) / (high.rolling(nd).max() - low.rolling(nd).min())
factors["rev_1d"] = -lr
factors["rev_1d_vs"] = -lr / lr.rolling(20).std()
factors["rev_2d"] = -(np.log(close) - np.log(close.shift(2)))
factors["rev_3d"] = -(np.log(close) - np.log(close.shift(3)))
factors["mom_10d_skip5"] = np.log(close / close.shift(10)) - np.log(close / close.shift(5))

# cross-sectional ranks per day
ranked = {}
for name, f in factors.items():
    r = f.rank(axis=1, pct=True)
    ranked[name] = r

# pooled pairwise spearman among ranked factors
names = list(ranked)
mat = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        x = ranked[a].stack().dropna()
        y = ranked[b].stack().dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < 100:
            mat.loc[a, b] = np.nan
        else:
            mat.loc[a, b] = spearmanr(x.loc[idx], y.loc[idx]).statistic
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print(mat.round(2))

# daily cross-sectional IC of each factor vs next-day return
ret = close.pct_change().shift(-1)
print("\n--- daily CS IC (last 250 days window if available) ---")
for name in names:
    ics = []
    for d in dates[:-1]:
        r = ranked[name].loc[d].dropna()
        y = ret.loc[d].dropna()
        idx = r.index.intersection(y.index)
        if len(idx) >= 5:
            ics.append(spearmanr(r.loc[idx], y.loc[idx]).statistic)
    ics = [x for x in ics if x == x]
    print(f"{name:16s} ic1={np.mean(ics):+.4f} n={len(ics)}")

# recent regime stats
print("\n--- recent 60d stats ---")
r60 = close.pct_change().tail(60)
print("mean daily market:", float(r60.mean(axis=1).mean()))
print("ann vol market:", float(r60.mean(axis=1).std() * np.sqrt(252)))
print("last 20d mean:", float(r60.tail(20).mean(axis=1).mean()))
print("per-asset 60d cum return:")
print((close.pct_change().tail(60) + 1).prod().round(3))
