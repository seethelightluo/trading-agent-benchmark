"""miner_2 2031-08-07 cycle: explore new factor candidates as the momentum ensemble decays.
Regime: risk-off / VIX-falling, WTI heavy pullback, ETH/BTC/haven leaders, yields flat.
Focus: cross-asset rotation (relative momentum), crypto/flight beta, haven-rotation beta, asymmetry.
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
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()

def assess(name, factor_df, show_regime=True):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:30s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
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

print("===== NEW CANDIDATES =====")

# A. Relative momentum: asset 20d return minus cross-sectional median 20d return (rotation breadth)
med20 = pd.DataFrame({s: retk(px[s],20) for s in WATCH}).median(axis=1)
f_relmom = build(pd.DataFrame({s: retk(px[s],20)-med20 for s in WATCH}))
assess('rel_mom_20_vs_med', f_relmom)

# B. Crypto-vs-defensive rotation: BTC-vs-XAU relative 20d momentum (risk appetite rotation)
btc_xau = retk(px['BTC'],20)-retk(px['XAU'],20)
f_rot = build(pd.DataFrame({s: retk(px[s],20) - btc_xau for s in WATCH}))
assess('mom20_align_crypto_havengap', f_rot)

# C. Relative to SPX: own 20d momentum minus SPX 20d momentum (equity-relative rotational)
f_rel = build(pd.DataFrame({s: retk(px[s],20)-retk(px['SPX'],20) for s in WATCH}))
assess('relmom20_vs_spx', f_rel)

# D. Vol-scaled momentum (sharpe-like: momentum per unit risk, favor smoother leaders)
f_sm = build(pd.DataFrame({s: retk(px[s],20)/rv(px[s],20) for s in WATCH}))
assess('sharpe_mom20_v20', f_sm)

# E. Asymmetry: upside vol vs downside vol 20d (flight-to-quality asymmetry)
def asym(s, win=20):
    v=vseries(s); r=v.pct_change()
    up=r.clip(lower=0).rolling(win).std(); dn=r.clip(upper=0).rolling(win).std()
    return (up-dn).reindex(INDEX)
f_asym = build(pd.DataFrame({s: asym(px[s]) for s in WATCH}))
assess('asym_updown_20', f_asym)

# F. GCR: 60d mean / 60d vol (rolling calmar-like quality)
def gcr(s, win=60):
    v=vseries(s); r=v.pct_change()
    return (v/v.shift(win)-1).reindex(INDEX)/r.rolling(win).std().reindex(INDEX)
f_gcr = build(pd.DataFrame({s: gcr(px[s]) for s in WATCH}))
assess('calmar60', f_gcr)

# G. Rotational beta to XAU (haven beta): 60d beta * sign of 10d XAU move
def beta_cond(p, reg, window=60, cond=10):
    a=retk(p,1); b=retk(reg,1)
    mb=retk(reg,cond).reindex(INDEX)
    beta=a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))
f_xaurot = build(pd.DataFrame({s: rot_cond(px[s], px['XAU']) for s in WATCH}))
assess('xau_beta_cond_60x10', f_xaurot)

print("\nDONE")