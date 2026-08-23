"""miner_2 2031-04-03: re-validate active effective factors through visible_through (2031-04-02).
Continuous re-validation / drift check on flip_mom_20x10, mom_diff_20_60, usdcny_beta_60.
Use shared miner_3_20261203_common helpers (15-asset cross-asset universe, no 50/80/300 requirement)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho)

ASOF = load_visible_through()
H = 10
px = load_prices(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} px_last={px.index[-1].date()} horiz={H}")
print(f"instruments: {len(WATCH)} | GATE |IC|>=0.0070, |ICIR|>=0.0840")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s)
    return (v.shift(-h)/v - 1.0).reindex(INDEX)
def flip_mom(p, kw=20, ks=10):
    return (retk(p, kw) * np.sign(retk(p, ks))).reindex(INDEX)
def mom_diff(p, kf=20, ks=60):
    return (retk(p, kf) - retk(p, ks)).reindex(INDEX)
def beta_to(p, regs, window=60):
    a = retk(p, 1); b = retk(regs, 1)
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    return (cov/var).reindex(INDEX)

def report(name, f):
    f = f.replace([np.inf,-np.inf], np.nan)
    fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
    icd = cross_sectional_ic(f, fwd)
    st = ic_stats(icd)
    if len(icd):
        ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)])
        ic180 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=180)])
        ic60 = ic_stats(icd.tail(60))
        ic30 = ic_stats(icd.tail(30)) if len(icd) >= 30 else None
    else:
        ic365 = ic180 = ic60 = ic30 = None
    cov_ic = (f.notna() & fwd.notna()).mean().mean()
    ad = float(f.notna().mean().mean())
    to = float(f.rank(axis=1).diff().abs().mean(axis=1).mean()) if len(f) else float('nan')
    print(f"\n==={name}===")
    print(f"FULL: IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',np.nan):.1f} cov_ad={ad:.3f} cov_ic={cov_ic:.3f} to={to:.3f}")
    if ic365 is not None: print(f"365d: IC={ic365['ic']:.4f} ICIR={ic365['icir']:.4f} n={ic365['n_dates']}")
    if ic180 is not None: print(f"180d: IC={ic180['ic']:.4f} ICIR={ic180['icir']:.4f} n={ic180['n_dates']}")
    if ic60 is not None: print(f"60d:  IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} n={ic60['n_dates']}")
    if ic30 is not None: print(f"30d:  IC={ic30['ic']:.4f} ICIR={ic30['icir']:.4f} n={ic30['n_dates']}")
    for lab, seg in regime_split(icd).items():
        print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
    dec = {}
    for hh in [1,3,5,10,20]:
        fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
        icd_h = cross_sectional_ic(f, fh)
        dec[hh] = float(icd_h['ic'].mean()) if len(icd_h) else float('nan')
    print(f"  decay: {dec}")
    return f

f_flip = report("flip_mom_20x10 (dir=1)", pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index())
f_momd = report("mom_diff_20_60 (dir=1)", pd.DataFrame({s: mom_diff(px[s]) for s in WATCH}).sort_index())

mac = load_macro(ASOF)
reg = mac['USDCNY'] if 'USDCNY' in mac.columns else pd.Series(dtype=float)
if len(reg) > 0:
    f_beta = pd.DataFrame({s: beta_to(px[s], reg) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
    fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
    icb = cross_sectional_ic(f_beta, fwd)
    sb = ic_stats(icb)
    print(f"\n===USDCNY_BETA_60 (dir=1)===")
    print(f"FULL: IC={sb['ic']:.4f} ICIR={sb['icir']:.4f} hit={sb['hit']:.3f} n={sb['n_dates']} avg_n={sb.get('avg_n',np.nan):.1f} cov={(f_beta.notna()&fwd.notna()).mean().mean():.3f}")
    if len(icb):
        cb60 = ic_stats(icb.tail(60))
        print(f"60d: IC={cb60['ic']:.4f} ICIR={cb60['icir']:.4f} n={cb60['n_dates']}")
        for lab, seg in regime_split(icb).items():
            print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
    print("corr flip_vs_beta:", round(spearman_panel_rho(f_flip, f_beta), 4))
    print("corr momdiff_vs_beta:", round(spearman_panel_rho(f_momd, f_beta), 4))

print("\ncorr flip_vs_momdiff:", round(spearman_panel_rho(f_flip, f_momd), 4))
print("\nDEPRECATE prox: full IC/ICIR gate + recent-window sign. Capture for updating persistence.")