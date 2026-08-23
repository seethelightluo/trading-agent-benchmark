"""miner_2 novel factor exploration at 2033-07-20 (visible through 2033-07-19).

Revalidation context: rel_mom/downside_vol/kurt are strongly NEGATIVE in the
recent year (momentum has flipped sign); beta_ew and max_ret remain positive and
pass gates. Goal: discover NEW low-correlation factors that pass
|IC|>=0.0070 & |ICIR|>=0.0840 @ h10 on recent 1y and (if possible) full window.
SCREEN ONLY (no persistence here). No lookahead.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2033-07-19"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
ep = 1e-9
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def full(name, f):
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    turn = rank_turnover(f, 10)
    s = ic.dropna()
    sub = s[s.index >= s.index.max() - np.timedelta64(365, "D")]
    ic_r = sub.mean() if len(sub) else np.nan
    icir_r = sub.mean()/sub.std(ddof=1) if len(sub)>2 and sub.std(ddof=1)>0 else np.nan
    gate_f = bool(abs(st["ic"])>=0.0070 and abs(st["icir"])>=0.0840)
    gate_r = bool(abs(ic_r)>=0.0070 and abs(icir_r)>=0.0840) if np.isfinite(ic_r) else False
    return dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                hit=round(st["hit"],3), n=st["n"],
                ic_1y=round(ic_r,4) if np.isfinite(ic_r) else None,
                icir_1y=round(icir_r,3) if np.isfinite(icir_r) else None,
                gate_full=gate_f, gate_1y=gate_r,
                covAD=round(cov["coverage_asset_days"],3), turn=round(turn,3),
                maxlib=round(corr,4))

cands = {}

def rolling_beta(x, y, win, mp=30):
    cov = x.rolling(win, min_periods=mp).cov(y)
    var = y.rolling(win, min_periods=mp).var()+ep
    return cov.divide(var, axis=0)

# 1. NEGATIVE cross-sectional 20d momentum (recent regime: momentum has flipped negative)
mom20 = close/close.shift(25)-1.0
cands["neg_rel_mom_20d_skip5"] = -(mom20.subtract(mom20.median(axis=1), axis=0))

# 2. NEGATIVE downside-volatility (recent regime negative)
r=ret
neg = r.where(r<0,0.0); ds=(neg**2).rolling(20).mean().apply(np.sqrt); tot=r.rolling(20).std()
cands["neg_downside_vol_ratio_20"] = (ds/tot)  # sign flipped vs active factor

# 3. conditional beta to US10Y (rate shock), directioned by 20d US10Y momentum
u = close["US10Y"]; u_r = u.pct_change(); u_mom = u/u.shift(20)-1.0
cands["us10y_beta_cond_60x20"] = rolling_beta(ret, u_r, 60).multiply(u_mom.clip(-0.5,0.5), axis=0)

# 4. conditional beta to VIX (fear), directioned by -20d VIX mom (falling VIX = calmer)
v = macro["VIX"]; v_r = v.pct_change(); v_mom = v/v.shift(20)-1.0
cands["vix_calm_cond_60x20"] = -rolling_beta(ret, v_r, 60).multiply(v_mom, axis=0)

# 5. momentum of downside-vol regime (acceleration of downside risk)
dn = (neg.diff(20).abs()).rolling(20).mean()
cands["downside_mom_20"] = -dn.rank(axis=1)

# 6. 20d high-minus-low / close (calm). recent signal suggests calmer assets outperformed?
cands["range_20_calm"] = (close.rolling(20).max()-close.rolling(20).min())/close

# 7. 60d drawdown depth (negated): less-drawn-down assets -> defensive
dd = (close/close.rolling(60).max()-1.0)
cands["drawdown_60"] = dd

# 8. cross-sectional 60d vol * 20d downside combined (total-risk regime)
vol60 = ret.rolling(60).std()
cands["risk_regime_60"] = -(vol60.rank(axis=1)) * (ds.rank(axis=1))

rows = []
for name, f in cands.items():
    rows.append(full(name, f))
res = pd.DataFrame(rows).sort_values("ic_1y", key=lambda s: s.fillna(0).abs(), ascending=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
print(res[["name","ic","icir","n","ic_1y","icir_1y","gate_full","gate_1y","covAD","turn","maxlib"]].to_string(index=False))
json.dump(rows, open("scripts/miner_2_20330720_explore.json","w"), indent=1)
print("\nsaved scripts/miner_2_20330720_explore.json")