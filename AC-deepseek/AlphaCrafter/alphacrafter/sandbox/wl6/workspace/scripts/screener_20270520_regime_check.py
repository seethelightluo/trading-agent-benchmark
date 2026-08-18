"""Screener-side regime/factor check (2027-05-20 cycle).

Reads only; computes current cross-sectional factor values and recent rank ICs
on the 15-asset benchmark. Does NOT run backtest/step and does NOT write to the
live account or date.json.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from factor_validation_lib import load_panel, load_macro, rank_ic_series, TRADABLE

pd.set_option("display.width", 200)

panel = load_panel("2027-05-19")  # visible through previous completed day
print("panel shape:", panel.shape, "last date:", panel.index[-1].date())

ret = panel.pct_change()

def mom(win, skip=5):
    return panel.shift(skip) / panel.shift(skip + win) - 1.0

def std_win(w, minp=10):
    return ret.rolling(w, min_periods=minp).std()

factors = {
    "mom_120d_skip5": mom(120),
    "mom_10d_skip5": mom(10),
    "vol_of_vol20x60": std_win(20).rolling(60).std(),
    "down_vol_ratio_20x120": -std_win(20).clip(lower=0) / std_win(120).clip(lower=0),
    "low_vol_20d": -std_win(20),
}
# vol_of_vol needs min_periods on the outer window too
factors["vol_of_vol20x60"] = std_win(20).rolling(60, min_periods=30).std()

# ---- 1) current cross-sectional values (last bar) ----
last = panel.index[-1]
print("\n=== current factor values @", last.date(), "===")
cur = {}
for fid, f in factors.items():
    cur[fid] = f.loc[last]
    print(f"\n{fid}:")
    print(cur[fid].sort_values(ascending=False).round(4).to_string())

# ---- 2) recent rank IC (last 130 trading days of factor history) ----
print("\n=== recent rank IC (last ~130d, horizon 10d) ===")
H = 10
fwd = panel.shift(-H) / panel - 1.0
start = panel.index[-131] if len(panel) > 131 else panel.index[0]
for fid, f in factors.items():
    sub_f = f.loc[start:last]
    sub_r = fwd.loc[start:last]
    ic = rank_ic_series(sub_f, sub_r, min_instr=8)
    ic = ic.dropna()
    if len(ic) == 0:
        print(f"{fid}: no data")
        continue
    icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
    print(f"{fid:24s} IC={ic.mean():+.4f} ICIR={icir:+.3f} hit={ (ic>0).mean():.3f} n={len(ic)}")

# ---- 3) regime stats ----
print("\n=== regime stats (last 60d) ===")
p60 = panel.tail(60)
r60 = p60.pct_change().dropna()
print("60d total return per asset:")
print((p60.iloc[-1] / p60.iloc[0] - 1).round(4).sort_values(ascending=False).to_string())
print("\n20d realized vol per asset (annualized-ish, daily std):")
print(r60.tail(20).std().round(4).sort_values(ascending=False).to_string())
print("\ncross-sectional dispersion (std of 60d returns):", round((p60.iloc[-1] / p60.iloc[0] - 1).std(), 4))
