"""
miner_2 novel factor exploration at 2031-08-11 (visible through 2031-08-08).
Test candidate families distinct from active library, full h10 IC/ICIR gate.
Admission: abs(IC)>=0.0070, abs(ICIR)>=0.0840. Report max_abs_library_corr.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, max_lib_corr)

END = "2031-08-08"
close = load_close(END)
macro = load_macro(END)
lib = library_panel(close, macro)
ret = close.pct_change()
fwd = forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def full(name, f):
    ic = daily_ic(f, fwd)
    st = ic_stats(ic, 10)
    cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    return dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                covAD=cov["coverage_asset_days"], corr=corr)

# ---- candidate constructions (cross-sectional, per-asset) ----
cands = {}

# A. Vol-scaled momentum: 20d return / 20d realized vol
vol20 = ret.rolling(20).std()
cands["vol_adj_mom_20"] = -ret.rolling(20).sum() / vol20  # short top

# B. 1d return autocorrelation (mean-reversion) over 20d, skip5
r_skip = ret.shift(5)
cands["autocorr_1x5_20"] = ret.rolling(20).corr(r_skip.shift(0))

# C. Drawdown from 60d high (reversal)
cands["drawdown_60d"] = - (close / close.rolling(60).max() - 1.0)

# D. Volume flow: 20d avg volume / 60d avg volume (using pct change as vol proxy if no vol)
# Use dollar-volume via |ret|*close as activity proxy
amp = (ret.abs() * close).rolling(20).mean() / (ret.abs() * close).rolling(60).mean()
cands["activity_flow_20x60"] = amp

# E. Realized-vol z-score: current 20d vol vs 120d mean vol (vol regime)
vol120 = ret.rolling(120).std()
cands["vol_z_20x120"] = -(vol20 - vol120) / vol120  # low vol = favorable

# F. Cross-sectional momentum composite (sign-weighted multi-window)
cands["mom_composite_5x10x60"] = (ret.rolling(5).sum() + ret.rolling(10).sum() + ret.rolling(60).sum()/6)

# G. Conditional beta to WTI (commodity) recent memory
wti_r = close["WTI"].pct_change()
cov = ret.rolling(60).cov(wti_r); var = wti_r.rolling(60).var()
cands["wti_beta_60d"] = cov.divide(var, axis=0)

# H. Conditional beta to XAU (gold) 
xau_r = close["XAU"].pct_change()
cov = ret.rolling(60).cov(xau_r); var = xau_r.rolling(60).var()
cands["xau_beta_60d"] = cov.divide(var, axis=0)

# I. 20d semi-downside deviation (magnitude of down moves) - distinct from ratio
neg = ret.where(ret < 0, 0.0)
cands["downside_dev_20"] = neg.rolling(20).std()

# J. Trend smoothness: |20d return| / sum(|daily ret|) efficiency, flip
cands["trend_efficiency_20"] = (ret.rolling(20).sum().abs() / ret.abs().rolling(20).sum())

rows = []
for name, f in cands.items():
    rows.append(full(name, f))

print(f"\n{'factor':28s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} {'covAD':>6s} {'maxLib':>6s} {'gate':>4s}")
for r in rows:
    gp = abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840
    print(f"{r['name']:28s} {r['ic']:7.4f} {r['icir']:7.3f} {r['hit']:5.2f} {r['n']:5d} "
          f"{r['covAD']:6.2f} {r['corr']:6.3f} {'PASS' if gp else ''}")