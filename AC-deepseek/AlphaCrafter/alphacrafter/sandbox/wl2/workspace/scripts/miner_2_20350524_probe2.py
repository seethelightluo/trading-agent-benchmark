"""miner_2 probe 2: check flat-feed history and effective moving cross-section."""
import pandas as pd, numpy as np

SYMS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = pd.Timestamp('2035-05-23')

def load_col(colname):
    cols = {}
    for s in SYMS:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        cols[s] = df.set_index('date')[colname]
    out = pd.DataFrame(cols).sort_index()
    return out[out.index <= END]

P = load_col('close')
R = P.pct_change()

# last date each symbol had |ret| > 1e-9
last_move = R.apply(lambda c: c[c.abs() > 1e-9].index.max() if (c.abs() > 1e-9).any() else pd.NaT)
print("=== last nonzero-return date per symbol ===")
for s in SYMS:
    print(f"{s:12s} {str(last_move[s])[:10]}  last_close={P[s].iloc[-1]:.4f}")

# how many dates since 2026-07-16 have zero cross-sectional movement (all 15 flat)?
flat_all = (R.abs() <= 1e-9).all(axis=1)
print("\n=== dates with ALL 15 names flat since 2026-07-16 ===")
sub = flat_all[flat_all.index >= '2026-07-16']
print("count:", sub.sum(), "of", len(sub))

# moving subset: 9 names?
MOVING = ['SPX','N225','000688.SH','SOX','NDX','XAU','COPPER','WTI','ETH']
Rm = R[MOVING]
flat_m = (Rm.abs() <= 1e-9).all(axis=1)
sub2 = flat_m[flat_m.index >= '2026-07-16']
print("dates where all 9 'moving' names flat:", sub2.sum())

# cross-sectional std of 10d returns through time (moving names only)
r10 = Rm.rolling(10).sum().iloc[-1]
print("\nlast 10d ret moving names:", r10.round(4).to_dict())

# full-sample 10d fwd return cross-sectional std - sanity on dispersion
fwd10 = (P.shift(-10)/P - 1)
cs_std = fwd10[MOVING].std(axis=1)
print("10d fwd cross-sectional std (moving): mean %.4f last %.4f" % (cs_std.mean(), cs_std.iloc[-1]))
