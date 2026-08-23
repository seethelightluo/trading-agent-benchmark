"""miner_2 novel factor exploration at 2032-01-09 (visible through 2032-01-08).
Focus on families with LOW correlation to the active 8-factor library which may
offer orthogonal alpha in the long-standing high-VIX sideways drift regime.
Gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 @ h10 (15-asset universe).
SCREEN ONLY (no persistence here)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr)

END = "2032-01-08"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def full(name, f):
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    return dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                hit=round(st["hit"],3), n=st["n"], covAD=round(cov["coverage_asset_days"],3),
                covD8=round(cov["coverage_dates_ge8"],3), maxlib=round(corr,4), top=pairs)

cands = {}
ep = 1e-9
# 1. Slow 120d momentum (long-trend) - explore direction as-is (prefer winners)
cands["mom_120d"] = close.shift(5)/close.shift(125)-1.0
# 2. Low realized vol 60d (defensive long, high-VIX drift regime)
cands["lowvol_60"] = -ret.rolling(60).std()
# 3. Drawdown-distance-to-peak 120d, positive = deep drawdown (mean-reversion long)
cands["dd_120"] = -(close/close.rolling(120).max()-1.0)
# 4. Skewness 60d (positive skew -> short in drift tape), flipped
cands["skew_60flip"] = -ret.rolling(60).skew()
# 5. Autocorrelation/reversal 1d (short recent pop)
cands["rev_1d"] = -ret
# 6. Trend efficiency: |net 40d move| / sum|abs moves| over 40d
net = (close/close.shift(40)-1.0).abs()
path = ret.abs().rolling(40).sum()+ep
cands["efratio_40"] = net/path
# 7. vol-of-vol ratio: 10d vol vs 60d vol (short vol-spike), flipped
cands["volratio_10x60"] = -(ret.rolling(10).std()/ret.rolling(60).std())
# 8. up/down semi-vol ratio 60d flipped (prefer lower upside) defensive
up = ret.where(ret>0,0.0); dwn = ret.where(ret<0,0.0)
cands["updownvol_60"] = -(up.rolling(60).std()+ep)/(dwn.abs().rolling(60).std()+ep)
# 9. Conditional beta to USDJPY trend (carry-like) win=60 cond=20
jpy_r = macro["USDJPY"].pct_change()
cov = ret.rolling(60).cov(jpy_r); var = jpy_r.rolling(60).var()+ep; beta = cov.divide(var, axis=0)
jpy_mom = macro["USDJPY"]/macro["USDJPY"].shift(20)-1.0
cands["jpy_beta_cond_60x20"] = -beta.multiply(jpy_mom, axis=0)
# 10. range position 60d (short overbought)
h = close.rolling(60).max(); l = close.rolling(60).min()
cands["rangepos_60"] = -((close-l)/(h-l))
# 11. 5d reversal (short recent pop) after momentum skip
cands["rev_5d"] = -(close/close.shift(5)-1.0)
# 12. max drawdown recovery: 250d drawdown from peak
cands["dd_250"] = -(close/close.rolling(250).max()-1.0)

rows = []
for name, f in cands.items():
    rows.append(full(name, f))
res = pd.DataFrame(rows).sort_values("ic", key=lambda s: s.abs(), ascending=False)
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
print(res.to_string(index=False))