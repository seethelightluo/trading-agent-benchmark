"""miner_2 novel factor exploration at 2032-05-17 (visible through 2032-05-14).
Recent regime: risk-off drift (-3.63% last block), momentum broke down.
Focus: defensive/cross-sectional volatility, downside-tail, rate-duration beta - families
LOW-correlated to the active 8-factor library, strong in the RECENT window.
Gate: abs(IC)>=0.0070 AND abs(ICIR)>=0.0840 @ h10. SCREEN ONLY, no persistence.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2032-05-14"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
ep = 1e-9
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def full(name, f, recent_from="2031-11-01"):
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    turn = rank_turnover(f, 10)
    s = ic.dropna(); sub = s[s.index >= recent_from]
    ic_r = sub.mean() if len(sub) else np.nan
    icir_r = sub.mean()/sub.std(ddof=1) if len(sub)>2 and sub.std(ddof=1)>0 else np.nan
    full_gate = abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840
    recent_gate = abs(ic_r) >= 0.0070 and abs(icir_r) >= 0.0840 if np.isfinite(icir_r) else False
    return dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                ic_r=round(ic_r,4) if np.isfinite(ic_r) else None,
                icir_r=round(icir_r,3) if np.isfinite(icir_r) else None,
                hit=round(st["hit"],3), n=st["n"], covAD=round(cov["coverage_asset_days"],3),
                turn=round(turn,3), maxlib=round(corr,4), fgate=bool(full_gate), rgate=bool(recent_gate),
                top=pairs)

cands = {}
# 1. cross-sectional realized downside vol 20d (defensive: high downside vol predicts LOWER fwd ret)
rv = (ret**2).rolling(20).mean().apply(np.sqrt)
ds = (ret.where(ret<0,0)**2).rolling(20).mean().apply(np.sqrt)
cands["downside_vol_20"] = ds                     # prefer low downside vol (positive-IC direction in raw sense)
cands["downside_vol_share_20"] = ds/(rv+ep)       # downside share / skew of vol

# 2. long/short vol regime ratio (vol acceleration: 10d vs 60d rolling vol)
vol10 = ret.rolling(10).std(); vol60 = ret.rolling(60).std()
cands["vol_accel_10x60"] = -(vol10/(vol60+ep))    # safety: low vol-accleration favored in risk-off
cands["vol_ratio_5x60"] = -(ret.rolling(5).std()/(ret.rolling(60).std()+ep))

# 3. conditional beta to VIX change (with sign so high factor = protective/open to risk)
vix = macro["VIX"]; vix_r = vix.pct_change()
vix_mom = vix/vix.shift(20)-1.0
cov = ret.rolling(60).cov(vix_r); var = vix_r.rolling(60).var()+ep
vix_beta = cov.divide(var, axis=0)
cands["vix_beta_neg"] = -vix_beta                 # low sensitivity to VIX-rise favored (defensive)
cands["vix_beta_cond_neg_up60"] = -vix_beta.multiply((vix/vix.shift(20)-1.0).clip(lower=0), axis=0)

# 4. rate-duration: beta to US10Y yield change, directioned by 20d yield move
us10y = close["US10Y"]; y_r = us10y.pct_change(); y_mom = us10y/us10y.shift(20)-1.0
cov = ret.rolling(60).cov(y_r); var = y_r.rolling(60).var()+ep
us10_beta = cov.divide(var, axis=0)
cands["us10_beta_cond_60x20"] = us10_beta.multiply(y_mom, axis=0)
cands["us10_beta_neg"] = -us10_beta

# 5. tail-distance: distance (# of own-stdev) below 1y high (mean-reversion in crash)
z = (close.rolling(252).max()-close)/(ret.rolling(252).std()+ep)
cands["tail_dist_252"] = z

# 6. cross-sectional return dispersion regime hedge: beta to equal-weight cross-sectional
#    volatility (compression implies follow-through risk)
cs_vol = ret.std(axis=1).rolling(20).mean()
cands["cs_vol_regime"] = 1.0  # placeholder, computed below

rows = []
for name, f in cands.items():
    if name == "cs_vol_regime":
        continue
    rows.append(full(name, f))

res = pd.DataFrame(rows).sort_values("ic_r", key=lambda s: s.abs(), ascending=False, na_position="last")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
print(res[["name","ic","icir","ic_r","icir_r","hit","n","covAD","turn","maxlib","fgate","rgate"]].to_string(index=False))
json.dump(rows, open("scripts/miner_2_20320517_explore.json","w"), indent=1)
print("\nsaved scripts/miner_2_20320517_explore.json")