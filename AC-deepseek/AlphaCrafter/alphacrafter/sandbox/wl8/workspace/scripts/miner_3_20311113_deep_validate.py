"""miner_3 2031-11-13 cycle: deep validation of realized_abs_ret_60 (neg) candidate
plus horizon-decay/turnover/correlation audit, and more per-asset factors to
ensure across-regime robustness. Output compact lines.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)


ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)}")

def vseries(s): return s.dropna()
def retk(s, k):
    return (vseries(s) / vseries(s).shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v - 1.0).reindex(INDEX)
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def abs_ret_mean(s, w=60):
    v = vseries(s); return v.pct_change().abs().rolling(w).mean().reindex(INDEX)

# NEGATIVE realized abs-return (low-vol quality/defensive): high abs-ret gets low score
f_ra = build(-pd.DataFrame({s: abs_ret_mean(px[s],60) for s in WATCH}))

print("===== HORIZON DECAY for realized_abs_ret_60 (neg) =====")
for h in (1,3,5,10,20):
    fwdh = pd.DataFrame({s: forward(px[s], h) for s in WATCH}).sort_index()
    icd = cross_sectional_ic(f_ra, fwdh)
    st = ic_stats(icd) if len(icd) else {'ic':np.nan,'icir':np.nan,'n_dates':0,'avg_n':np.nan}
    print(f"  h={h:2d} IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} n={st['n_dates']}")

print("===== regime robustness @h10 =====")
fwd10 = pd.DataFrame({s: forward(px[s],10) for s in WATCH}).sort_index()
icd = cross_sectional_ic(f_ra, fwd10)
for lab, seg in regime_split(icd).items():
    if seg[2]: print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
for dlab, days in [('365d',365),('120d',120)]:
    rm = icd.index >= icd.index[-1]-pd.Timedelta(days=days)
    if rm.any():
        s=ic_stats(icd[rm]); print(f"  {dlab} IC={s['ic']:+.4f} ICIR={s['icir']:+.4f} n={s['n_dates']}")

print("===== library correlation (report only, gate recomputes) =====")
def skip5_mom(s, k):
    v=vseries(s); r=v.pct_change(5).rolling(k//5).mean(); return r.reindex(INDEX)
f_mom10 = build(pd.DataFrame({s: skip5_mom(px[s],10) for s in WATCH}))
def beta_cond(p, reg, window=60, cond=20):
    a=retk(p,1); b=retk(reg,1); mb=retk(reg,cond)
    beta=a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))
f_vb = build(pd.DataFrame({s: beta_cond(px[s], mac['VIX']) for s in WATCH}))
f_yb = build(pd.DataFrame({s: beta_cond(px[s], px['US10Y']) for s in WATCH}))
for nm, other in [('mom_10d_skip5',f_mom10), ('vix_beta_cond_60x20',f_vb), ('yield_beta_cond_60x20',f_yb)]:
    rho = spearman_panel_rho(f_ra, other.astype(float))
    print(f"  rho with {nm}: {rho:+.3f}")

print("===== more per-asset candidates @h10 =====")
f_abs10 = build(-pd.DataFrame({s: abs_ret_mean(px[s],10) for s in WATCH}))
st=ic_stats(cross_sectional_ic(f_abs10, fwd10))
print(f"  realized_abs_ret_10(neg) IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} n={st['n_dates']}")

cross20 = pd.concat([retk(px[z],20) for z in WATCH], axis=1).mean(axis=1)
f_res = build(pd.DataFrame({s: retk(px[s],20)-cross20 for s in WATCH}))
st2=ic_stats(cross_sectional_ic(f_res, fwd10))
print(f"  resid_mom_20 IC={st2['ic']:+.4f} ICIR={st2['icir']:+.4f} n={st2['n_dates']}")

# turnover for candidate
rank = f_ra.rank(axis=1, pct=True)
to = rank.diff().abs().mean().mean()
print(f"  realized_abs_ret_60 neg mean rank-turnover={to:.4f}")
print("\nDONE deep_validate")