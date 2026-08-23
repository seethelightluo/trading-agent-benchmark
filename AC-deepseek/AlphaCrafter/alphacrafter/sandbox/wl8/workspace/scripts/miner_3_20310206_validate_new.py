"""miner_3 2031-02-06: validate two new gate-passing candidates for persistence.
Candidates: sharp_mom20 (risk-adjusted momentum), vol_low_20 (negative 20d vol).
Compute admission metrics: IC, ICIR, coverage, turnover, decay, library correlation.
Persist any that pass gate AND have low library correlation.
"""
import sys, os, json
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

# ---- reconstruct library factors for correlation ----
def beta_cond(p, reg, window=60, cond=20):
    a = retk(p,1); b = retk(reg,1)
    mb = retk(reg, cond).reindex(INDEX)
    cov = a.rolling(window).cov(b); var = b.rolling(window).var()
    beta = cov/var
    return build(beta * np.sign(mb))

f_mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
f_vix   = build(pd.DataFrame({s: beta_cond(px[s], mac['VIX']) for s in WATCH}))
f_yield = build(pd.DataFrame({s: beta_cond(px[s], px['US10Y']) for s in WATCH}))
f_flip  = build(pd.DataFrame({s: retk(px[s],20)*np.sign(retk(px[s],10)) for s in WATCH}))
f_momdd = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))
lib = {'flip_mom_20x10': f_flip, 'mom_diff_20_60': f_momdd,
       'mom_10d_skip5': f_mom10, 'vix_beta_cond_60x20': f_vix,
       'yield_beta_cond_60x20': f_yield}

def worth(name, fd):
    icd = cross_sectional_ic(fd, fwd)
    st = ic_stats(icd)
    cov = (fd.notna() & fwd.notna()).mean().mean()
    rank = fd.rank(axis=1)
    to = rank.diff().abs().mean().mean()
    maxrho = 0.0; details = {}
    for lname, lf in lib.items():
        r = spearman_panel_rho(fd, lf)
        details[lname] = r if np.isfinite(r) else 0.0
        maxrho = max(maxrho, abs(details[lname]))
    print(f"=== {name} ===")
    print(f"  FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',np.nan):.1f} cov={cov:.3f} turnover={to:.4f}")
    for lab, seg in regime_split(icd).items():
        print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    print(f"  max_abs_library_corr={maxrho:.4f}")
    for k,v in details.items(): print(f"    rho vs {k}: {v:.4f}")
    admitted = (abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840)
    print(f"  GATE: |IC|={abs(st['ic']):.4f}>=0.007? {abs(st['ic'])>=0.007} | |ICIR|={abs(st['icir']):.4f}>=0.084? {abs(st['icir'])>=0.084} -> ADMITTED={admitted}")
    return {'st': st, 'cov': cov, 'to': to, 'maxrho': maxrho, 'rho': details, 'admitted': admitted}

cands = {}
# sharp_mom20: mom20 / annualized vol20 (risk-adjusted momentum)
vol20 = build(pd.DataFrame({s: rv(px[s],20) for s in WATCH}))
mom20 = build(pd.DataFrame({s: retk(px[s],20) for s in WATCH}))
cands['sharp_mom20'] = build(mom20 / (vol20*np.sqrt(252)).replace(0,np.nan))
# vol_low_20: negative 20d realized vol (low-vol tilt)
cands['vol_low_20'] = -vol20

results = {}
for name, fd in cands.items():
    st, cov, to, maxrho, rho, admitted = worth(name, fd)
    results[name] = dict(st=st, cov=cov, to=to, maxrho=maxrho, rho=rho, admitted=admitted)

print("\n=== SUMMARY ===")
for k, r in results.items():
    print(f"{k}: ADMITTED={r['admitted']} IC={r['st']['ic']:+.4f} ICIR={r['st']['icir']:+.4f} maxrho={r['maxrho']:.4f} cov={r['cov']:.3f}")