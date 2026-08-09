"""Miner2 family exploration: cross-asset beta & macro-sensitivity factors.
Idea: relative sensitivity of each tradable asset to a global equity factor,
to DXY, to VIX, and to UST moves may rank assets for next-day returns
(e.g., high-DXY-beta assets vs low; risk-on/off tilt).
Vectorized on common-date panel. Research window ends 2026-07-15.
"""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
from miner2_fast import fwd_returns, fast_ic, screen, turnover10, coverage_panel, ic_all

closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx) for s in SYMBOLS}).astype(float)
RET = CP.pct_change()
print(f"common dates={len(idx)} {idx.min().date()}..{idx.max().date()}")

# forward returns
fwd1 = fwd_returns(closes, 1).reindex(idx)
fwd5 = fwd_returns(closes, 5).reindex(idx)
fwd10 = fwd_returns(closes, 10).reindex(idx)
n_cells = len(idx) * len(SYMBOLS)

# ---- macro series aligned to common index ----
MAC = {}
macro_dir = "../persistent/index_data"
for m in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
    d = pd.read_csv(os.path.join(macro_dir, f"{m}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= idx.max()].set_index("date").sort_index()
    MAC[m] = d["close"].reindex(idx)

EW = RET.mean(axis=1)  # equal-weight cross-asset factor
SPXR = RET["SPX"]
MAC["EW"] = EW
MACR = pd.DataFrame({k: v.pct_change() for k, v in MAC.items() if k in MAC})
MACR["EW"] = EW

# ---- rolling beta helpers ----
def roll_beta(y, x, win):
    """rolling beta of y on x using simple covariance/var (both aligned)."""
    m = pd.concat([y, x], axis=1).dropna()
    yv = m.iloc[:, 0]; xv = m.iloc[:, 1]
    cov = yv.rolling(win).cov(xv)
    var = xv.rolling(win).var()
    return (cov / var).reindex(idx)

def roll_beta_panel(xr, win, name_prefix=None, sign=1.0):
    cols = {}
    for s in SYMBOLS:
        b = roll_beta(RET[s], xr, win)
        cols[s] = b * sign
    return pd.DataFrame(cols)

panels = {}

# 1) beta to global equity (EW) and to SPX
for win in (20, 60, 120):
    panels[f"beta_ew_{win}"] = roll_beta_panel(EW, win)
    panels[f"beta_spx_{win}"] = roll_beta_panel(SPXR, win)

# 2) beta to DXY, VIX, US10Y, USDJPY
for m in ["DXY", "VIX", "USDJPY"]:
    xr = MACR[m]
    for win in (20, 60, 120):
        panels[f"beta_{m}_{win}"] = roll_beta_panel(xr, win)

# 3) DXY beta times DXY direction (conditional USD regime) --- interaction
dxy_ret = MACR["DXY"]
dxy_trend = dxy_ret.rolling(60).mean()
# 4) VIX sensitivity conditional on VIX level
vix_lvl = MAC["VIX"]

results = []
for name, panel in panels.items():
    results.append(screen(name, panel, fwd1, fwd5, fwd10, n_cells))

print("\n--- PASS gate check ---")
for r in results:
    if r["passed"]:
        print(r["name"], r["ic1"])

print(f"\nfamily done | {sum(r['passed'] for r in results)} passed / {len(results)}")