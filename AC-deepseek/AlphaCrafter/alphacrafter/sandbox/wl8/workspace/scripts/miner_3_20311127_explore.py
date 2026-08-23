"""miner_3 2031-11-27: re-validate active/fallback ensemble factors + explore new
per-asset candidate constructs through visible_through 2031-11-26.

Regime (memory 2031-11-27): high-vol divergent regime, flat block; winners WTI/BTC/SPX/US10Y,
drags ETH/COPPER/N225/XAU. Goal: fresh uncorrelated alpha beyond pure momentum.
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
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)} px_last={px.index[-1].date()}")

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

def assess(name, factor_df):
    icd = cross_sectional_ic(factor_df, fwd)
    if len(icd)==0:
        print(f"{name:28s} NO DATES (broadcast/constant)"); return None, None
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:28s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',np.nan):4.1f} cov={cov:.3f}")
    rmask = icd.index >= icd.index[-1]-pd.Timedelta(days=365)
    if rmask.any():
        ic365 = ic_stats(icd[rmask]); line += f" | 365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f}"
    g = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    line += f" | {'PASS' if g else 'FAIL'}"
    print(line)
    for lab, seg in regime_split(icd).items():
        print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st, icd

print("===== REVALIDATE FALLBACK ENSEMBLE (as-of gates) =====")
def skip5_mom(s, k):
    v = vseries(s); r = v.pct_change(5).rolling(k//5).mean(); return r.reindex(INDEX)
f_mom10 = build(pd.DataFrame({s: skip5_mom(px[s],10) for s in WATCH}))
assess('mom_10d_skip5 (+)', f_mom10)
def beta_cond(p, reg, window=60, cond=20):
    a = retk(p,1); b = retk(reg,1)
    mb = retk(reg, cond).reindex(INDEX)
    beta = a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))
f_vb = build(pd.DataFrame({s: beta_cond(px[s], mac['VIX']) for s in WATCH}))
assess('vix_beta_cond_60x20', f_vb)
f_yb = build(pd.DataFrame({s: beta_cond(px[s], px['US10Y']) for s in WATCH}))
assess('yield_beta_cond_60x20', f_yb)

print("\n===== NEW PER-ASSET CANDIDATES (h=10) =====")
# low realized vol (neg abs ret), shorter window
def abs_ret(s, w):
    v = vseries(s); return v.pct_change().abs().rolling(w).mean().reindex(INDEX)
f_av60 = build(pd.DataFrame({s: -abs_ret(px[s],60) for s in WATCH}))
assess('realized_abs_ret_60 (neg)', f_av60)
# distance below 60d high (drawdown/bounce)
def dhigh(s, win):
    v = vseries(s); return (v / v.rolling(win).max() - 1.0).reindex(INDEX)
f_dh60 = build(pd.DataFrame({s: dhigh(px[s],60) for s in WATCH}))
assess('dist_high_60', f_dh60)
# price position in 120d range (pos_in_range / mean-reversion)
def pos_range(s, win):
    v = vseries(s); r = v.rolling(win).max()-v.rolling(win).min()
    return ((v-v.rolling(win).min())/r).reindex(INDEX)
f_pos120 = build(pd.DataFrame({s: pos_range(px[s],120) for s in WATCH}))
assess('pos_in_120range', f_pos120)
# short reversal: -5d return
f_rev5 = build(pd.DataFrame({s: -retk(px[s],5) for s in WATCH}))
assess('rev_5d (-)', f_rev5)
# 20d momentum (pure)
f_m20 = build(pd.DataFrame({s: retk(px[s],20) for s in WATCH}))
assess('mom_20d', f_m20)
# cross-asset residual momentum 10d
cross10 = pd.concat([retk(px[z],10) for z in WATCH], axis=1).mean(axis=1)
f_res = build(pd.DataFrame({s: retk(px[s],10)-cross10 for s in WATCH}))
assess('resid_mom_10', f_res)
# XAU-COPPER relative (haven vs cycle) - ONLY as mutable composite, robust
fxau = retk(px['XAU'],20).reindex(INDEX); fcop = retk(px['COPPER'],20).reindex(INDEX)
f_hc = build(pd.DataFrame({s: (fxau-fcop).copy() for s in WATCH}))
assess('xau_minus_cop_20', f_hc)
# volat ratio short/long (regime spread)
f_vr = build(pd.DataFrame({s: rv(px[s],5)/rv(px[s],60) for s in WATCH}))
assess('vol_ratio_5x60', f_vr)
print("\nDONE")