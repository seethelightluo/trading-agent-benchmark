"""miner_2 explore candidate factors at 2032-03-22 (data through 2032-03-20).
Batch of orthogonal candidates (momentum horizon variant, vol-ratio, drawdown,
skew, weekly reversal, RSI). Reports full + recent_4m split, library corr.
Gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 @ h10.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2032-03-20"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
ep = 1e-9
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

cands = {}
# 1. 40d momentum (skip 5) - different horizon than active 20d
cands["mom_40d_skip5"] = close.shift(5)/close.shift(45)-1.0
# 2. 60d momentum (skip 5) - slower trend
cands["mom_60d_skip5"] = close.shift(5)/close.shift(65)-1.0
# 3. relative vol spike 10x60 (short)
cands["vol_ratio_10x60"] = -(ret.rolling(10).std()/ret.rolling(60).std())
# 4. 60d drawdown (mean reversion: favor deeper drawdown)
cands["dd_60"] = -(close/close.rolling(60).max()-1.0)
# 5. skew 60d (short positive skew in drift tape)
cands["skew_60d"] = -ret.rolling(60).skew()
# 6. weekly reversal 5d (short recent pop)
cands["rev_5d"] = -(close/close.shift(5)-1.0)
# 7. RSI 14 (short overbought, mean reversion)
delta = ret
up = delta.where(delta>0,0.0).rolling(14).mean(); dn = (-delta.where(delta<0,0.0)).rolling(14).mean()
rs = (up+ep)/(dn+ep); rsi = 100-100/(1+rs)
cands["rsi14_short"] = -rsi
# 8. trend MA slope ratio: faster MA vs slower MA (5/20)
cands["ma_slope_5x20"] = (close/close.rolling(20).mean())/(close.rolling(5).mean()/close.rolling(20).mean()).rolling(1).mean()
# simpler: price vs 20MA normalized by 60d vol (z-score of price vs MA)
cands["gain_20v60"] = (close-close.rolling(60).mean())/close.rolling(60).std()

res_rows = []
for name, f in cands.items():
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    ic_s = ic.dropna()
    sub = ic_s[ic_s.index >= "2031-11-01"]
    ic_r = sub.mean() if len(sub) else np.nan
    icir_r = sub.mean()/sub.std(ddof=1) if len(sub)>2 and sub.std(ddof=1)>0 else np.nan
    res_rows.append(dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                         hit=round(st["hit"],3), n=st["n"],
                         ic_r=round(ic_r,4), icir_r=round(icir_r,3),
                         cov=round(cov["coverage_asset_days"],3),
                         maxlib=round(corr,4), top=pairs))

res = pd.DataFrame(res_rows).sort_values("ic", key=lambda s: s.abs(), ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 40)
print(res.to_string(index=False))
json.dump(res_rows, open("scripts/miner_2_20320322_explore.json","w"), indent=1)
print("\nsaved scripts/miner_2_20320322_explore.json")