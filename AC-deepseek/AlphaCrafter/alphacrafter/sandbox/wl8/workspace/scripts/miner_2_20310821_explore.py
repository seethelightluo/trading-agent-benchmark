"""miner_2 2031-08-21 cycle: explore new factor candidates as of visible_through 2031-08-20.

Regime: high-vol risk-on fading; VIX still ~47; commodities/energy (COPPER/XAU/WTI)
lead while equities/crypto pulled back; defensives flat. Focus on rotation/
flight-quality/asymmetry/regime-dispersion constructs unlikely to crowd the
fallback mom10/vix/yield trio.
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
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)} px_last={px.index[-1].date()}\n")

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

def assess(name, factor_df, show_regime=True, show_recent=True):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:32s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',np.nan):4.1f} cov={cov:.3f}")
    rmask = icd.index >= icd.index[-1]-pd.Timedelta(days=365)
    if show_recent and rmask.any():
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

# A. Relative momentum vs cross-sectional median (rotation breadth, own vs median)
med20 = pd.DataFrame({s: retk(px[s],20) for s in WATCH}).median(axis=1)
f_relmom = build(pd.DataFrame({s: retk(px[s],20)-med20 for s in WATCH}))
assess('rel_mom_20_vs_med', f_relmom)

# B. Vol-scaled momentum 20 (Sharpe-like, favor smooth leaders)
f_sm = build(pd.DataFrame({s: retk(px[s],20)/rv(px[s],20) for s in WATCH}))
assess('sharpe_mom20_v20', f_sm)

# C. GCR / calmar 60 (quality: return per unit risk)
def gcr(s, win=60):
    v=vseries(s); r=v.pct_change()
    return (v/v.shift(win)-1).reindex(INDEX)/r.rolling(win).std().reindex(INDEX)
f_gcr = build(pd.DataFrame({s: gcr(px[s]) for s in WATCH}))
assess('calmar60', f_gcr)

# D. Up/down asymmetry 20d (flight-to-quality asymmetry)
def asym(s, win=20):
    v=vseries(s); r=v.pct_change()
    up=r.clip(lower=0).rolling(win).std(); dn=r.clip(upper=0).rolling(win).std()
    return (up-dn).reindex(INDEX)
f_asym = build(pd.DataFrame({s: asym(px[s]) for s in WATCH}))
assess('asym_updown_20', f_asym)

# E. Downside capture ratio 60d (bad-day dominance)
def down_up_capture(s, win=60):
    v=vseries(s); r=v.pct_change()
    dn=r.clip(upper=0).rolling(win).sum(); up=r.clip(lower=0).rolling(win).sum()
    return (dn/up).reindex(INDEX)
f_duc = build(pd.DataFrame({s: down_up_capture(px[s]) for s in WATCH}))
assess('down_up_capture_60', f_duc)

# F. XAU-haven rotation beta: 60d beta to XAU, conditioned on 10d XAU move
def beta_cond(p, reg, window=60, cond=10):
    a=retk(p,1); b=retk(reg,1)
    mb=retk(reg,cond).reindex(INDEX)
    beta=a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))
f_xaurot = build(pd.DataFrame({s: beta_cond(px[s], px['XAU']) for s in WATCH}))
assess('xau_rot_beta_60x10', f_xaurot)

# G. Drawdown proximity / max drawdown over 60d (mean-reversion of deep underperformers)
def maxdd60(s):
    v=vseries(s); running=(v/v.cummax()-1.0).rolling(60).min()
    return running.reindex(INDEX)
f_mdd = build(pd.DataFrame({s: maxdd60(px[s]) for s in WATCH}))
assess('maxdd_60', -f_mdd)

print("\nDONE")