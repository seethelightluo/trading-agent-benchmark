"""miner_3 2031-08-07 cycle: re-validate active factors + explore new candidate
constructs through visible_through (2031-08-06). Continuous re-validation cycle.

Regime context (see memory): ongoing risk-off / VIX-falling rotation; heavy WTI
pullback; ETH still largest winner; 4/15 frozen breadth names (000688/SOX/NDX/CN10Y).
Explore cross-asset/microstructure/yield-beta candidates unlikely to crowd the
pure-momentum ensemble (flip_mom_20x10, mom_diff_20_60).
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
def mv(s, win):
    return vseries(s).rolling(win).mean().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def assess(name, factor_df, show_regime=True):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:26s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
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

# ===== 0. Re-validate active ensemble factors =====
print("===== REVALIDATE ACTIVE ENSEMBLE FACTORS =====")
f_flip = build(pd.DataFrame({s: retk(px[s],20)*np.sign(retk(px[s],10)) for s in WATCH}))
assess('flip_mom_20x10', -f_flip)  # persisted direction: need check
f_momdd = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))
assess('mom_diff_20_60', -f_momdd)

print("\n===== NEW CANDIDATES (cross-asset / microstructure / yield) =====")
# A. 60d realized-vol z-score (relative vol regime, low = defensive quality)
f_volz = build(pd.DataFrame({s: -((rv(px[s],20)-rv(px[s],60).rolling(120).mean())/rv(px[s],60).rolling(120).std()) for s in WATCH}))
assess('vol_z_20_vs_60', f_volz)

# B. downside vs upside capture over 60d (bad days matter more in risk-off)
def down_up_capture(s, win=60):
    v = vseries(s); r = v.pct_change()
    down = r.clip(upper=0).rolling(win).sum()
    up = r.clip(lower=0).rolling(win).sum()
    return (down/up).reindex(INDEX)
f_duc = build(pd.DataFrame({s: down_up_capture(px[s]) for s in WATCH}))
assess('down_up_capture_60', f_duc)

# C. yield-beta conditioned on yield trend (rates-risk rotation)
def beta_cond(p, reg, window=60, cond=20):
    a = retk(p,1); b = retk(reg,1)
    mb = retk(reg, cond).reindex(INDEX)
    beta = a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))
f_yb = build(pd.DataFrame({s: beta_cond(px[s], px['US10Y']) for s in WATCH}))
assess('yld_beta_cond_60x20', f_yb)

# D. DXY-beta conditioned on DXY trend (dollar regime, USDCNY already handled)
f_db = build(pd.DataFrame({s: beta_cond(px[s], mac['DXY']) for s in WATCH}))
assess('dxy_beta_cond_60x20', f_db)

# E. range proximity 20d (mean reversion in whipsaw)
def pos_in_range(s, win):
    v=vseries(s); r=v.rolling(win).max()-v.rolling(win).min()
    return ((v-v.rolling(win).min())/r).reindex(INDEX)
f_pos20 = build(pd.DataFrame({s: -pos_in_range(px[s],20) for s in WATCH}))
assess('rev_range_pos_20', f_pos20)

# F. 10d vol of vol (volatility acceleration)
def vol_of_vol(s, a=20, b=60):
    v=vseries(s); rv_=r.pct_change(); return rv_.rolling(a).std().reindex(INDEX) if False else (rv(s,a)-rv(s,b))
f_vov = build(pd.DataFrame({s: -(rv(px[s],20)-rv(px[s],60)) for s in WATCH}))
assess('vol_accel_20x60', f_vov)

print("\nDONE")