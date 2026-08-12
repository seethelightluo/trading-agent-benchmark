"""miner_2 2029-02-08: data probe + current regime assessment (read-only)."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    GRID, ASSETS, asset_series, to_grid, load_macro, cross_sectional_rank,
    spearman_ic_matrix, fwd_by_horizon_dict, HORIZON, MIN_ASSETS,
)

series = asset_series()
print("assets with data:", sorted(series.keys()))
print("n_assets:", len(series), "n_grid_dates:", len(GRID))
print("grid start:", GRID[0], "grid end:", GRID[-1])

# Current regime stats (last 5, 20, 60 trading days)
print("\n=== Current regime (through visible date) ===")
for s, df in series.items():
    c = df["close"]
    r5 = c.iloc[-1] / c.iloc[-6] - 1 if len(c) > 6 else np.nan
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    v20 = df["ret"].iloc[-20:].std() * np.sqrt(252)
    print(f"{s:10s} px={c.iloc[-1]:12.2f} r5={r5:+.3f} r20={r20:+.3f} r60={r60:+.3f} vol20_ann={v20:.3f}")

# Macro observations
print("\n=== Macro ===")
for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    s = load_macro(m)
    if s is None:
        print(m, "MISSING")
        continue
    v = s.dropna()
    if len(v) < 25:
        print(m, "too short", len(v))
        continue
    print(f"{m:8s} last={v.iloc[-1]:.2f} r5={v.iloc[-1]/v.iloc[-6]-1:+.3f} r20={v.iloc[-1]/v.iloc[-21]-1:+.3f}")

# Cross-sectional dispersion of 20d returns
rets = {}
for s, df in series.items():
    rets[s] = df["close"].iloc[-1] / df["close"].iloc[-21] - 1
rs = np.array(list(rets.values()))
print("\n20d cross-sectional: mean={:.3f} std={:.3f} max={:.3f} min={:.3f}".format(rs.mean(), rs.std(), rs.max(), rs.min()))

# Recent IC environment of simple momentum (sanity check that cross-section is informative)
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
mom20 = to_grid({s: (df["close"].shift(5) / df["close"].shift(25) - 1.0) for s, df in series.items()})
ics = spearman_ic_matrix(cross_sectional_rank(mom20), fwd10)
idx = np.array([t for t, _ in ics]); icv = np.array([v for _, v in ics])
print("\nmom20 sanity: n={} ic={:+.4f} icir={:+.4f} hit={:.3f}".format(len(icv), icv.mean(), icv.mean()/icv.std(), (icv>0).mean()))
m = idx >= len(GRID) - 250
print("mom20 last250: ic={:+.4f} icir={:+.4f} n={}".format(icv[m].mean(), icv[m].mean()/icv[m].std(), m.sum()))
