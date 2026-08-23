"""miner_3 2031-05-15: screen new candidate factors suited to the current
high-vol / VIX-falling / risk-off rotation regime. Validate IC/ICIR through 2031-05-14.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s); return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v - 1.0).reindex(INDEX)
def rv(s, win):
    v = vseries(s); return v.pct_change().rolling(win).std().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()

def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)
def assess(name, factor_df, show_60=True, show_regime=True):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    gate = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    ic365=ic_stats(icd[icd.index>=icd.index[-1]-pd.Timedelta(days=365)])
    ic180=ic_stats(icd[icd.index>=icd.index[-1]-pd.Timedelta(days=180)])
    ic60=ic_stats(icd.tail(60))
    line = (f"{name:24s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',np.nan):4.1f} cov={cov:.3f} | "
            f"365d {ic365.get('ic',np.nan):+.4f}/{ic365.get('icir',np.nan):+.4f} "
            f"180d {ic180.get('ic',np.nan):+.4f}/{ic180.get('icir',np.nan):+.4f} "
            f"60d {ic60.get('ic',np.nan):+.4f}/{ic60.get('icir',np.nan):+.4f} | {'PASS' if gate else 'FAIL'}")
    print(line)
    if show_regime:
        for lab, seg in regime_split(icd).items():
            print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st, icd

vix = mac['VIX']

print("\n===== VIx regime / risk-off candidates =====")
# VIX level z-score conditioning (zero low) - scale each asset by 1 if VIX rising
# A. volatility ratio: short-term(5) / long-term(60) realized vol
f_volr = build(pd.DataFrame({s: rv(px[s],5)/rv(px[s],60) for s in WATCH}))
assess('volratio_5x60', f_volr)
# B. VIX slope x cross-asset momentum
vixslope = mac['VIX'].pct_change().rolling(10).mean()
f_vixslope = build(pd.DataFrame({s: retk(px[s],20)*np.sign(vixslope) for s in WATCH}))
assess('mom20_x_vixslope', f_vixslope)
# C. 10d momentum (short horizon continuation)
f_mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
assess('mom_10d', f_mom10)
# D. max 5d drawdown recovery proxy: -5d return (mean reversion)
f_rev5 = build(pd.DataFrame({s: -retk(px[s],5) for s in WATCH}))
assess('rev_5d', f_rev5)
# E. distance-from-high at 60d (bounce candidate in rotation)
def dfromhigh(s, win):
    v=vseries(s); return (v / v.rolling(win).max() - 1.0).reindex(INDEX)
f_dh = build(pd.DataFrame({s: dfromhigh(px[s],60) for s in WATCH}))
assess('dist_high_60', f_dh)
# F. 120d low-rebound strength: (close-120low)/(120range)
def pos_in_range(s, win):
    v=vseries(s); r=v.rolling(win).max()-v.rolling(win).min()
    return ((v-v.rolling(win).min())/r).reindex(INDEX)
f_pos = build(pd.DataFrame({s: pos_in_range(px[s],120) for s in WATCH}))
assess('pos_in_120range', f_pos)
# G. momentum 20d conditioned on VIX < median (low-vol regime continuation)
vixmed = vix.rolling(120).median().reindex(INDEX)
vi = vix.reindex(INDEX); vm = vix.rolling(120).median().reindex(INDEX); f_momlv = build(pd.DataFrame({s: retk(px[s],20)*(vi < vm).astype(float) for s in WATCH}))
assess('mom20_lowvix', f_momlv)

print("\nDONE")