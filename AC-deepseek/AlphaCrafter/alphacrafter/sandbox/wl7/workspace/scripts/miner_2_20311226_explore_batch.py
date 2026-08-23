"""miner_2 novel factor exploration batch at 2031-12-26 (visible through 2031-12-25).
Focus on families with LOW correlation to the active 8-factor library.
Motivated by current sideways-to-bear drift regime: drawdown/distance-to-peak,
slow mean-reversion, long-window momentum, low-vol.
Gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 @ h10 (15-asset universe).

Explores ONE theme per candidate but multiple variants in one batch (screen only,
no persistence here).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr)

END = "2031-12-25"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

out_rows = []
def full(name, f):
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    return dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                hit=round(st["hit"],3), n=st["n"],
                covAD=round(cov["coverage_asset_days"],3),
                maxlib=round(corr,4), top=pairs)

cands = {}
# farthest-distance-to-peak (drawdown) over various windows; negate -> prefer assets far from peak (mean-reversion)
for w in (60, 120, 250):
    c = -(close / close.rolling(w).max() - 1.0)   # positive = drawdown deep
    cands[f"dd_{w}"] = c

# closeness to peak AFTER drawdown already recovered -- actually use raw -distance to peak flip
# long momentum windows
for w, skip in ((30,5),(60,5),(90,5),(120,10)):
    cands[f"mom{w}s{skip}"] = close.shift(skip)/close.shift(w+skip)-1.0

# slow price position in 60/120 range (oscillator), negate short overbought
for w in (60,120):
    h = close.rolling(w).max(); l = close.rolling(w).min()
    cands[f"range_pos_{w}"] = -((close-l)/(h-l))

# low-vol factor (short 30d realized vol), rank cross-sectionally
rv30 = ret.rolling(30).std()
cands["lowvol_30"] = -rv30

# drawdown recovery speed: return from DD trough over 60d
# (current - 60d_min)/|60d_min|

# 5d return reversal (short recent pop) - mean reversion
cands["rev_5d"] = -(close/close.shift(5)-1.0)

# volatility-of-return directional: 10d vol vs 60d vol ratio (short relative vol spike)
cands["vol_ratio_10x60"] = -(ret.rolling(10).std()/ret.rolling(60).std())

# skewness 60d (high positive skew bearish in drift tape)
cands["skew_60d"] = -ret.rolling(60).skew()

# autcorrelation/reversal 1d
cands["rev_spot"] = -ret

for name, f in cands.items():
    out_rows.append(full(name, f))

res = pd.DataFrame(out_rows).sort_values("ic", key=lambda s: s.abs(), ascending=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False))