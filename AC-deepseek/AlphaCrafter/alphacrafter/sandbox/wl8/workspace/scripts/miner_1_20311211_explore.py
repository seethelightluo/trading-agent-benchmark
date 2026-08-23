"""miner_1 2031-12-11: explore fresh uncorrelated factor candidates beyond the
fallback momentum/beta ensemble. Library is empty (0/30), all previously evicted.
Regime (memory 2031-12-11): high-vol divergent regime, VIX ~41, winners WTI/BTC/SPX/US10Y,
drags ETH(-29%)/COPPER/N225/XAU. Seek robust, regime-independent, interpretable signals.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)} px_last={px.index[-1].date()}")
print(f"mac cols={list(mac.columns)}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s); return (v/v.shift(k)-1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v-1.0).reindex(INDEX)
def rv(s, win):
    v = vseries(s); return v.pct_change().rolling(win).std().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def assess(name, factor_df, gate_ic=0.0070, gate_icir=0.0840):
    icd = cross_sectional_ic(factor_df, fwd)
    if len(icd)==0:
        print(f"{name:30s} NO DATES (broadcast/constant)"); return None, None
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    turn = (factor_df.rank(axis=1).diff().abs().mean().mean()) if factor_df.shape[1]>1 else 0
    line = (f"{name:30s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',np.nan):4.1f} cov={cov:.3f} turn={turn:.3f}")
    rmask = icd.index >= icd.index[-1]-pd.Timedelta(days=365)
    if rmask.any():
        ic365 = ic_stats(icd[rmask]); line += f" | 365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f}"
    g = abs(st['ic'])>=gate_ic and abs(st['icir'])>=gate_icir
    line += f" | {'PASS' if g else 'FAIL'}"
    print(line)
    for lab, seg in regime_split(icd).items():
        print(f"    {lab}: IC={seg[0]:+.4f} ICIR={seg[1]:+.4f} n={seg[2]}")
    return st, icd

print("\n===== BASELINE (fallback ensemble) =====")
def skip5_mom(s, k):
    v=vseries(s); r=v.pct_change(5).rolling(k//5).mean(); return r.reindex(INDEX)
f_mom10 = build(pd.DataFrame({s: skip5_mom(px[s],10) for s in WATCH}))
assess('mom_10d_skip5 (+)', f_mom10)

print("\n===== NEW CANDIDATES (h=10) =====")
def abs_ret(s,w):
    v=vseries(s); return v.pct_change().abs().rolling(w).mean().reindex(INDEX)
# low-vol defensive short window
f_av20 = build(pd.DataFrame({s:-abs_ret(px[s],20) for s in WATCH}))
assess('realized_abs_ret_20 neg', f_av20)

# distance from 20d high (momentum continuation vs bounce)
def dhigh(s,win):
    v=vseries(s); return (v/v.rolling(win).max()-1.0).reindex(INDEX)
f_dh20 = build(pd.DataFrame({s:dhigh(px[s],20) for s in WATCH}))
assess('dist_high_20 (pos)', f_dh20)

# XAU/COPPER cycle spread applied as cross-sectional style tilt
fxau=retk(px['XAU'],20).reindex(INDEX); fcop=retk(px['COPPER'],20).reindex(INDEX)
f_hc = build(pd.DataFrame({s:(fxau-fcop).copy() for s in WATCH}))
assess('xau_minus_cop_20 (broadcast)', f_hc)

# cross-asset residual short-term reversal in 3d
cross3 = pd.concat([retk(px[z],3) for z in WATCH],axis=1).mean(axis=1)
f_res3 = build(pd.DataFrame({s:-(retk(px[s],3)-cross3) for s in WATCH}))
assess('resid_mom_3 neg', f_res3)

# vol ratio 5/60 (regime-vol expansion > bearish)
f_vr = build(pd.DataFrame({s:rv(px[s],5)/rv(px[s],60) for s in WATCH}))
assess('vol_ratio_5x60 (neg)', -f_vr)

# skew: (close-open)/(2*range)*window avg - candle skew mean reversion
def candle_skew(s,w):
    v=px[s]  # using level series
    return np.nan
f_sk = build(pd.DataFrame({s: rv(px[s],20) for s in WATCH}))
assess('(debug) rv20', f_sk)

print("\nDONE")