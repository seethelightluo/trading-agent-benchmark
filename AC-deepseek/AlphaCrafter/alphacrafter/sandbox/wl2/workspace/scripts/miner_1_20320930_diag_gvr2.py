"""miner_1 2032-09-30 deeper diagnostic: why gold_vs_rate_60 has 0 IC obs."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, load_asset, to_grid, spearman_ic_matrix,
    HORIZON, MIN_ASSETS, GRID, N_GRID,
)

DAYS = 4200
series = {}
for s in ASSETS:
    df = load_asset(s, days=DAYS)
    if df is None or len(df) < 120:
        continue
    close = df["close"].astype(float)
    series[s] = pd.DataFrame({"close": close, "ret": close.pct_change()})

xau = series["XAU"]["close"]
us10y = series["US10Y"]["close"]
xau_ret60 = xau / xau.shift(60) - 1.0
us10y_chg60 = us10y - us10y.shift(60)
spread_asset = xau_ret60 - us10y_chg60

panel = {}
for s, df in series.items():
    if s == "XAU":
        panel[s] = pd.Series(np.nan, index=df.index)
    else:
        panel[s] = spread_asset.reindex(df.index)

mat = to_grid(panel)
fwd10 = to_grid({s: df["close"].shift(-HORIZON) / df["close"] - 1.0 for s, df in series.items()})
print("mat shape:", mat.shape, "nan frac: %.3f" % np.isnan(mat).mean(), flush=True)
print("fwd nan frac: %.3f" % np.isnan(fwd10).mean(), flush=True)

# per-date valid pair counts
valid_pairs = (~np.isnan(mat) & ~np.isnan(fwd10)).sum(axis=1)
print("valid pairs: min=%d max=%d mean=%.2f" % (valid_pairs.min(), valid_pairs.max(), valid_pairs.mean()), flush=True)
print("dates with >=8 valid pairs:", int((valid_pairs >= MIN_ASSETS).sum()), "of", N_GRID, flush=True)

# show a mid-sample date row
t = 1500
print("\nsample row t=%d date=%s" % (t, GRID[t]), flush=True)
for j, s in enumerate(ASSETS):
    print("  %-10s factor=%+.4f fwd=%+.4f" % (s, mat[t, j], fwd10[t, j]), flush=True)

# check tail rows (maybe issue only in recent window?)
for t in [N_GRID - 30, N_GRID - 10, N_GRID - 1]:
    vp = valid_pairs[t]
    print("\nt=%d date=%s valid_pairs=%d" % (t, GRID[t], vp), flush=True)
    for j, s in enumerate(ASSETS):
        if np.isfinite(mat[t, j]):
            print("  %-10s factor=%+.4f fwd=%+.4f" % (s, mat[t, j], fwd10[t, j]), flush=True)
