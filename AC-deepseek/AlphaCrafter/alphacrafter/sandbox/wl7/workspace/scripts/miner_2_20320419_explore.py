"""miner_2 novel factor exploration at 2032-04-19 (visible through 2032-04-16).
Focus: orthogonal families with LOW correlation to the active 8-factor library,
valid in the ongoing high-VIX sideways-to-bear drift regime.
Gate: abs(IC)>=0.0070 AND abs(ICIR)>=0.0840 @ h10 (15-asset universe).
SCREEN ONLY (no persistence here). No lookahead.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2032-04-16"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
ep = 1e-9
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def full(name, f):
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    turn = rank_turnover(f, 10)
    ic_s = ic.dropna()
    sub = ic_s[ic_s.index >= "2031-11-01"]
    ic_r = sub.mean() if len(sub) else np.nan
    icir_r = sub.mean()/sub.std(ddof=1) if len(sub)>2 and sub.std(ddof=1)>0 else np.nan
    return dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                hit=round(st["hit"],3), n=st["n"], ic_r=round(ic_r,4) if np.isfinite(ic_r) else None,
                icir_r=round(icir_r,3) if np.isfinite(icir_r) else None,
                covAD=round(cov["coverage_asset_days"],3), turn=round(turn,3),
                maxlib=round(corr,4), top=pairs)

cands = {}
# 1. conditional beta to US10Y yield change (rate-beta, directioned by 20d yield momentum)
us10 = macro if False else None
us10y = close["US10Y"]
y_r = us10y.pct_change()
y_mom = us10y / us10y.shift(20) - 1.0
cov = ret.rolling(60).cov(y_r); var = y_r.rolling(60).var()+ep
us10_beta = cov.divide(var, axis=0)
cands["us10y_beta_cond_60x20"] = us10_beta.multiply(y_mom, axis=0)

# 2. conditional beta to VIX *change* (only when VIX rising - risk-off exposure), 60x20
vix = macro["VIX"]
vix_r = vix.pct_change(); vix_up = (vix/vix.shift(20)-1.0).where(vix/vix.shift(20)-1.0>0, 0.0)
cov = ret.rolling(60).cov(vix_r); var = vix_r.rolling(60).var()+ep
vix_beta = cov.divide(var, axis=0)
cands["vix_beta_cond_up_60x20"] = -vix_beta.multiply(vix_up, axis=0)

# 3. USDCNY conditional beta (CNY move), 60x20
cny = macro["USDCNY"]
cny_r = cny.pct_change(); cny_mom = cny/cny.shift(20)-1.0
cov = ret.rolling(60).cov(cny_r); var = cny_r.rolling(60).var()+ep
cny_beta = cov.divide(var, axis=0)
cands["cny_beta_cond_60x20"] = cny_beta.multiply(cny_mom, axis=0)

# 4. cross-sectional autocorrelation of returns at 5d lag (trend persistence)
cands["autocorr_5d"] = ret.rolling(40).apply(lambda x: pd.Series(x).autocorr(5) if len(x)>10 else np.nan, raw=False)

# 5. gap/price reversal: overnight-range efficiency - distance below 20d high favored (mean reversion) - dd_20
cands["dd_20"] = -(close/close.rolling(20).max()-1.0)

# 6. convexity/upside share: ratio of positive moves to total moves 20d (breadth of upside)
up_share = (ret>0).rolling(20).mean().astype(float)
cands["up_share_20"] = up_share

# 7. high-watermark: fraction of window spent below max (time under water, mean reversion)
def time_under_water(x):
    r = pd.Series(x)
    return (r < r.expanding().max()).mean()
cands["time_under_water_120"] = close.rolling(120).apply(lambda x: time_under_water((x/x.iloc[0])), raw=False)

rows = []
for name, f in cands.items():
    rows.append(full(name, f))

res = pd.DataFrame(rows).sort_values("ic", key=lambda s: s.abs(), ascending=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 50)
print(res.to_string(index=False))
json.dump(rows, open("scripts/miner_2_20320419_explore.json","w"), indent=1)
print("\nsaved scripts/miner_2_20320419_explore.json")