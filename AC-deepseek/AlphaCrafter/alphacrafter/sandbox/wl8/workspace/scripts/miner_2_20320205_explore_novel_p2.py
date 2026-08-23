"""miner_2 2032-02-05 Part 2: More novel factor ideas + momentum of momentum
"""
import sys, os, json, math
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho, zscore_series)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s)
    return (v.shift(-h) / v - 1.0).reindex(INDEX)
def rv(s, win):
    v = vseries(s)
    return v.pct_change().rolling(win).std().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()

def build(df):
    return df.sort_index().replace([np.inf, -np.inf], np.nan).astype(float)

def assess(name, factor_df, show_regimes=True):
    icd = cross_sectional_ic(factor_df, fwd)
    if len(icd) == 0:
        print(f"{name:35s} NO DATES"); return None, icd
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:35s} FULL IC={st['ic']:+.6f} ICIR={st['icir']:+.6f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',0):4.1f} cov={cov:.3f}")
    rmask = icd.index >= icd.index[-1] - pd.Timedelta(days=365)
    if rmask.any():
        ic365 = ic_stats(icd[rmask])
        line += f" | 365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f}"
    gate = abs(st['ic']) >= 0.0070 and abs(st['icir']) >= 0.0840
    line += f" | {'PASS' if gate else 'FAIL'}"
    print(line)
    if show_regimes:
        for lab, seg in regime_split(icd).items():
            print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st, icd

print("=" * 80)
print("IDEA 5: Distance from multi-period high (bounce/reversal signal)")
print("=" * 80)
for w in [20, 60, 120]:
    f = pd.DataFrame({s: (vseries(px[s]) / vseries(px[s]).rolling(w).max() - 1.0).reindex(INDEX) for s in WATCH})
    assess(f'dist_high_{w}d', build(f))

print("=" * 80)
print("IDEA 6: Distance from multi-period low (momentum/breakout)")
print("=" * 80)
for w in [20, 60, 120]:
    f = pd.DataFrame({s: (vseries(px[s]) / vseries(px[s]).rolling(w).min() - 1.0).reindex(INDEX) for s in WATCH})
    assess(f'dist_low_{w}d', build(f))

print("=" * 80)
print("IDEA 7: Momentum of momentum (acceleration) - 2nd derivative of price")
print("=" * 80)
def mom_mom(s, fast=10, slow=20):
    v = vseries(s)
    r_fast = (v / v.shift(fast) - 1.0)
    r_slow = (v / v.shift(slow) - 1.0)
    return (r_fast - r_slow).reindex(INDEX)

for fst, slw in [(5, 10), (10, 20), (10, 60), (20, 60)]:
    ff = pd.DataFrame({s: mom_mom(px[s], fst, slw) for s in WATCH})
    assess(f'mom_of_mom_{fst}x{slw}', build(ff))

print("=" * 80)
print("IDEA 8: Skewness/return asymmetry (negative skew predicts downside)")
print("=" * 80)
def rolling_skew(s, w):
    v = vseries(s)
    return v.pct_change().rolling(w).skew().reindex(INDEX)

for w in [20, 60, 120]:
    f_sk = pd.DataFrame({s: rolling_skew(px[s], w) for s in WATCH})
    assess(f'ret_skew_{w}d', build(f_sk))

print("=" * 80)
print("IDEA 9: Macro beta to DXY (USD strength factor)")
print("=" * 80)
dxy = mac['DXY']
for w in [20, 60]:
    dxy_ret = retk(dxy, 1)
    betas = {}
    for s in WATCH:
        a_ret = retk(px[s], 1)
        beta_rolling = a_ret.rolling(w).cov(dxy_ret) / dxy_ret.rolling(w).var()
        betas[s] = beta_rolling.reindex(INDEX)
    f_dxy_beta = build(pd.DataFrame(betas))
    assess(f'dxy_beta_{w}d', f_dxy_beta)

print("=" * 80)
print("IDEA 10: Cross-asset momentum breadth (average z-score)")
print("=" * 80)
for w in [5, 10, 20]:
    raw = pd.DataFrame({s: retk(px[s], w) for s in WATCH})
    z = raw.apply(lambda col: (col - col.median()) / (col.std() + 1e-8), axis=0)
    breadth = build(z.mean(axis=1).to_frame(name='breadth') for _ in WATCH)
    # Create panel: same breadth value for all assets on each date
    breadth_panel = build(pd.DataFrame({s: z.mean(axis=1) for s in WATCH}))
    assess(f'mom_breadth_z_{w}d', breadth_panel)

print("=" * 80)
print("IDEA 11: Composite volatility-zscore momentum (conditional)")
print("=" * 80)
for mw in [20]:
    mom_raw = pd.DataFrame({s: retk(px[s], mw) for s in WATCH})
    for vw in [60, 120]:
        vol_raw = pd.DataFrame({s: rv(px[s], vw) for s in WATCH})
        # Low-vol regime: favor momentum; high-vol regime: fade
        vol_z = vol_raw.apply(lambda col: (col - col.median()) / (col.std() + 1e-8), axis=0)
        wgt = build(np.exp(-vol_z.abs()))  # weight from 0.37 to 1.0
        f_cond = build(mom_raw * wgt)
        assess(f'vol_cond_mom_{mw}d_v{vw}d', f_cond)

# =======================================================
# IDEA 12: Re-investigate the current fallback ensemble factors 
# with the latest data to check if they still pass
# =======================================================
print("\n" + "=" * 80)
print("REVALIDATION OF CURRENT FALLBACK ENSEMBLE FACTORS (latest data)")
print("=" * 80)

def skip5_mom(s, k):
    v = vseries(s)
    r = v.pct_change(5).rolling(k // 5).mean()
    return r.reindex(INDEX)

f_mom10 = build(pd.DataFrame({s: skip5_mom(px[s], 10) for s in WATCH}))
assess('mom_10d_skip5 (fallback)', f_mom10)

def beta_cond(p, reg, window=60, cond=20):
    a = retk(p, 1)
    b = retk(reg, 1)
    mb = retk(reg, cond).reindex(INDEX)
    beta = a.rolling(window).cov(b) / b.rolling(window).var()
    return build(beta * n