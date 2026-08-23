"""miner_2 novel factor exploration at 2032-09-07 (visible through 2032-09-03).

Context from 20320906 revalidation: on full 15-asset window only beta_ew_60d
still passes gates (IC 0.0374/ICIR 0.099); rel_mom recovered in recent 1y
(+0.0332/+0.109) while downside_vol_ratio badly decayed (-0.0507/-0.167).
Goal: discover NEW low-correlation factors that pass |IC|>=0.0070 &
|ICIR|>=0.0840 @ h10 on both full-window and recent-1y.
SCREEN ONLY (no persistence here). No lookahead.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2032-09-03"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
ep = 1e-9
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def full(name, f):
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    turn = rank_turnover(f, 10)
    s = ic.dropna()
    sub = s[s.index >= "2031-09-01"]
    ic_r = sub.mean() if len(sub) else np.nan
    icir_r = sub.mean()/sub.std(ddof=1) if len(sub)>2 and sub.std(ddof=1)>0 else np.nan
    gate_f = bool(abs(st["ic"])>=0.0070 and abs(st["icir"])>=0.0840)
    gate_r = bool((abs(ic_r)>=0.0070 and abs(icir_r)>=0.0840)) if np.isfinite(ic_r) else False
    return dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                hit=round(st["hit"],3), n=st["n"],
                ic_r=round(ic_r,4) if np.isfinite(ic_r) else None,
                icir_r=round(icir_r,3) if np.isfinite(icir_r) else None,
                gate_full=gate_f, gate_recent=gate_r,
                covAD=round(cov["coverage_asset_days"],3), turn=round(turn,3),
                maxlib=round(corr,4), top=pairs)

cands = {}

def rolling_beta(x, y, win):
    cov = x.rolling(win).cov(y); var = y.rolling(win).var()+ep
    return cov.divide(var, axis=0)

# 1. cross-sectional skewness of returns, 20d, skip5 (tail asymmetry)
r = ret.shift(5)
cands["skew_20d_skip5"] = r.rolling(20, min_periods=12).skew()

# 2. conditional beta to WTI (energy risk), directioned by 20d WTI momentum
wti = close["WTI"]; wti_r = wti.pct_change(); wti_mom = wti/wti.shift(20)-1.0
cands["wti_beta_cond_60x20"] = rolling_beta(ret, wti_r, 60).multiply(wti_mom, axis=0)

# 3. conditional beta to XAU (flight-to-safety / gold), directioned by 20d XAU mom
xau = close["XAU"]; xau_r = xau.pct_change(); xau_mom = xau/xau.shift(20)-1.0
cands["xau_beta_cond_60x20"] = rolling_beta(ret, xau_r, 60).multiply(xau_mom, axis=0)

# 4. vol slope: short-term vol minus long-term vol (rising vol = bearish) -> negative
vol_5 = ret.rolling(5).std(); vol_60 = ret.rolling(60).std()
cands["vol_slope_20x60"] = -(vol_5 - vol_60)

# 5. inverse 20d realized vol (low-vol defensive, negative direction before rank)
cands["inv_vol_20"] = -ret.rolling(20).std()

# 6. momentum of realized vol (acceleration of risk)
vol_20 = ret.rolling(20).std()
cands["vol_mom_20"] = -(vol_20 / vol_20.shift(20) - 1.0)

# 7. 20d range efficiency (high-low)/close — calm factor
cands["range_20"] = -( (close.rolling(20).max()-close.rolling(20).min())/close )

# 8. conditional beta to N225 benchmark (Asia equity beta), directioned by 20d N225 mom
n225 = close["N225"]; n225_r = n225.pct_change(); n225_mom = n225/n225.shift(20)-1.0
cands["n225_beta_cond_60x20"] = rolling_beta(ret, n225_r, 60).multiply(n225_mom, axis=0)

rows = []
for name, f in cands.items():
    rows.append(full(name, f))

res = pd.DataFrame(rows).sort_values("ic", key=lambda s: s.abs(), ascending=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
print(res[["name","ic","icir","hit","n","ic_r","icir_r","gate_full","gate_recent","covAD","turn","maxlib"]].to_string(index=False))
json.dump(rows, open("scripts/miner_2_20320907_explore.json","w"), indent=1)
print("\nsaved scripts/miner_2_20320907_explore.json")
