"""miner_2 novel factor exploration visible through 2034-04-12.
Screens a batch of new low-library-correlation factors for the 15-asset universe.
Admission gates: |IC|>=0.0070 & |ICIR|>=0.0840 @ h10. SCREEN ONLY.
No lookahead. Reports full-window and recent-1y gates plus max library correlation.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2034-04-12"
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
    icir_r = sub.mean()/sub.std(ddof=1) if len(sub) > 2 and sub.std(ddof=1) > 0 else np.nan
    hit_r = float((sub > 0).mean()) if len(sub) else np.nan
    gate_f = bool(abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840)
    gate_r = bool(abs(ic_r) >= 0.0070 and abs(icir_r) >= 0.0840) if np.isfinite(ic_r) else False
    return dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                hit=round(st["hit"],3), n=st["n"], covAD=round(cov["coverage_asset_days"],3),
                ic_1y=round(ic_r,4) if np.isfinite(ic_r) else None,
                icir_1y=round(icir_r,3) if np.isfinite(icir_r) else None,
                hit_1y=hit_r if isinstance(hit_r, float) else None,
                gate_full=gate_f, gate_1y=gate_r, turn=round(turn,3), maxlib=round(corr,4))

def rolling_beta(x, y, win, mp=30):
    cov = x.rolling(win, min_periods=mp).cov(y)
    var = y.rolling(win, min_periods=mp).var() + ep
    return cov.divide(var, axis=0)

cands = {}

# 1. Conditional beta to CN10Y (China yield), directioned by 20d CN10Y momentum
c10 = close["CN10Y"]; c10_r = c10.pct_change(); c10_mom = c10/c10.shift(20)-1.0
cands["cn10y_beta_cond_60x20"] = rolling_beta(ret, c10_r, 60).multiply(c10_mom.clip(-0.5,0.5), axis=0)

# 2. Conditional beta to WTI (commodity cycle), directioned by 20d WTI momentum
wt = close["WTI"]; wt_r = wt.pct_change(); wt_mom = wt/wt.shift(20)-1.0
cands["wti_beta_cond_60x20"] = rolling_beta(ret, wt_r, 60).multiply(wt_mom.clip(-0.5,0.5), axis=0)

# 3. Autocorrelation of returns (trending vs mean-reverting), 10d
ac10 = ret.rolling(10).apply(lambda x: x.autocorr() if len(x) > 2 else np.nan, raw=False)
cands["autocorr_10d"] = ac10.rank(axis=1)

# 4. Efficiency ratio 20d (net move / total path) - trend quality
eff = (close/close.shift(20)-1).abs() / (ret.abs().rolling(20).sum())
cands["eff_ratio_20"] = eff.rank(axis=1)

# 5. Relative rank spread in 20d momentum (cross-sectional divergence)
rmom20 = (close/close.shift(25)-1.0).rank(axis=1)
cands["rel_rank_spread_20"] = rmom20 - rmom20.median(axis=1)

# 6. Skew 20d skipped 5 (recent regime re-check)
sk = ret.rolling(20, min_periods=12).skew().shift(5)
cands["skew_20d_skip5"] = sk.rank(axis=1)

# 7. Drawdown depth 20d (calm / defensive)
dd20 = (close.rolling(20).max()-close)/close.rolling(20).max()
cands["drawdown_20"] = dd20.rank(axis=1)

# 8. Downside-volatility acceleration (more downside concentration recently)
neg = ret.where(ret < 0, 0.0)
ds = (neg**2).rolling(20).mean().apply(np.sqrt)
ds_mom = ds / ds.shift(20)
cands["downside_mom_20"] = -ds_mom.rank(axis=1)

rows = []
for name, f in cands.items():
    rows.append(full(name, f))
res = pd.DataFrame(rows).sort_values("ic_1y", key=lambda s: s.fillna(0).abs(), ascending=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
print(res[["name","ic","icir","n","ic_1y","icir_1y","hit_1y","gate_full","gate_1y","covAD","turn","maxlib"]].to_string(index=False))
json.dump(rows, open("scripts/miner_2_20340412_explore.json","w"), indent=1)
print("\nsaved scripts/miner_2_20340412_explore.json")