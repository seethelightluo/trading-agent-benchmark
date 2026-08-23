"""miner_3 2031-11-13 cycle: re-validate active ensemble factors + explore new
candidate constructs through visible_through (2031-11-12).

Regime (from memory 2031-11-13): high-vol divergent regime; recent block -2.05%;
winners WTI/BTC/SPX/US10Y, drags ETH/COPPER/N225/XAU. 4/15 frozen breadth names
(000688/SOX/NDX/CN10Y). Goal: find fresh uncorrelated alpha beyond pure momentum.
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

def assess(name, factor_df, show_regime=True):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:28s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',np.nan):4.1f} cov={cov:.3f}")
    rmask = icd.index >= icd.index[-1]-pd.Timedelta(days=365)
    if rmask.any():
        ic365 = ic_stats(icd[rmask])
        line += f" | 365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f}"
    g = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    line += f" | {'PASS' if g else 'FAIL'}"
    print(line)
    if show_regime:
        for lab, seg in regime_split(icd).items():
            print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st, icd

print("===== REVALIDATE FALLBACK ENSEMBLE FACTORS (as-of library gates) =====")
# flip_mom_20x10: sign-flipped momentum product
f_flip = build(pd.DataFrame({s: retk(px[s],20)*np.sign(retk(px[s],10)) for s in WATCH}))
assess('flip_mom_20x10 (-)', -f_flip)
f_momdd = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))
assess('mom_diff_20_60 (-)', -f_momdd)
# fallback ensemble factors mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20
def skip5_mom(s, k):
    v = vseries(s); r = v.pct_change(5).rolling(k//5).mean(); return r.reindex(INDEX)
f_mom10 = build(pd.DataFrame({s: -skip5_mom(px[s],10) for s in WATCH}))
assess('mom_10d_skip5 (-)', f_mom10)
def beta_cond(p, reg, window=60, cond=20):
    a = retk(p,1); b = retk(reg,1)
    mb = retk(reg, cond).reindex(INDEX)
    beta = a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))
f_vb = build(pd.DataFrame({s: beta_cond(px[s], mac['VIX']) for s in WATCH}))
assess('vix_beta_cond_60x20', f_vb)
f_yb = build(pd.DataFrame({s: beta_cond(px[s], px['US10Y']) for s in WATCH}))
assess('yield_beta_cond_60x20', f_yb)

print("\n===== NEW CANDIDATES =====")
# N1. 10d normalized volume momentum (demand breadth across assets)
def vol_mom10(s):
    v=vseries(s); vc=v.rolling(10).mean(); return (vc/v.rolling(60).mean()-1).reindex(INDEX)
f_vmom = build(pd.DataFrame({s: vol_mom10(px[s]) for s in WATCH}))
assess('vol_ratio_10x60', f_vmom)

# N2. XAU vs COPPER relative trend (haven vs cyclical rotation)
if 'XAU' in WATCH and 'COPPER' in WATCH:
    fxau = retk(px['XAU'],20).reindex(INDEX); fcop = retk(px['COPPER'],20).reindex(INDEX)
    f_hc = build(pd.DataFrame({s: (fxau-fcop).copy() for s in WATCH}))
    assess('xau_minus_cop_20', f_hc)

# N3. crypto vs equity momentum differential (risk regime rotation)
if 'BTC' in WATCH and 'SPX' in WATCH:
    f_crv = build(pd.DataFrame({s: (retk(px['BTC'],10)-retk(px['SPX'],10)).copy() for s in WATCH}))
    assess('btc_minus_spx_10', f_crv)

# N4. 20d max drawdown (drawdown-mean-reversion / quality)
def drawdown20(s):
    v=vseries(s); roll=v.rolling(20).max(); return (v/roll-1).reindex(INDEX)
f_dd = build(pd.DataFrame({s: -drawdown20(px[s]) for s in WATCH}))
assess('rev_20d_drawdown', f_dd)

# N5. price vs 80d moving average distance (trend-strength)
def dist_ma(s, w):
    v=vseries(s); return (v/v.rolling(w).mean()-1).reindex(INDEX)
f_dma = build(pd.DataFrame({s: dist_ma(px[s],80) for s in WATCH}))
assess('trend_80d_ma_dist', f_dma)

print("\nDONE")