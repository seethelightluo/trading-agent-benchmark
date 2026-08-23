"""miner_2 2031-05-15: revalidate active library factors (flip_mom_20x10, mom_diff_20_60)
through visible_through (2031-05-14). No lookahead: factor uses data <= t, fwd uses t..t+h."""
import json, numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     zscore_series, spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} px_last={px.index[-1].date()} instruments={len(WATCH)}")

H = 10

def retk(s, k):
    v = s.dropna()
    return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = s.dropna()
    return (v.shift(-h)/v - 1.0).reindex(INDEX)
def flip_mom(p, kw=20, ks=10):
    return (retk(p, kw) * np.sign(retk(p, ks))).reindex(INDEX)
def mom_diff(p, kf=20, ks=60):
    return (retk(p, kf) - retk(p, ks)).reindex(INDEX)
def hsvol(p, k=20):
    return (retk(p, 1)).rolling(k).std().reindex(INDEX)

def report(name, f):
    fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
    icd = cross_sectional_ic(f, fwd)
    st = ic_stats(icd)
    print(f"\n=== {name} ===")
    print(f"  FULL: IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',np.nan):.1f}")
    if len(icd):
        ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)])
        ic180 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=180)])
        ic60 = ic_stats(icd.tail(60))
        ic30 = ic_stats(icd.tail(30)) if len(icd) >= 30 else None
        print(f"  365d: IC={ic365['ic']:.4f} ICIR={ic365['icir']:.4f} n={ic365['n_dates']}")
        print(f"  180d: IC={ic180['ic']:.4f} ICIR={ic180['icir']:.4f} n={ic180['n_dates']}")
        print(f"  60d : IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} n={ic60['n_dates']}")
        if ic30 is not None:
            print(f"  30d : IC={ic30['ic']:.4f} ICIR={ic30['icir']:.4f} n={ic30['n_dates']}")
        # sign stability check
        full_sign = np.sign(st['ic'])
        flip360 = np.sign(ic365['ic']) != full_sign
        flip180 = np.sign(ic180['ic']) != full_sign
        flip60 = np.sign(ic60['ic']) != full_sign
        print(f"  SIGN-FLIP check (vs FULL sign {int(full_sign):+d}): 365d={flip360} 180d={flip180} 60d={flip60}")
    for lab, seg in regime_split(icd).items():
        print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
    dec = {}
    for hh in [1,3,5,10,20]:
        fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
        icd_h = cross_sectional_ic(f, fh)
        dec[hh] = float(icd_h['ic'].mean()) if len(icd_h) else float('nan')
    print(f"  decay: {dec}")
    return f

f_flip = report("flip_mom_20x10", pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index())
f_momd = report("mom_diff_20_60", pd.DataFrame({s: mom_diff(px[s]) for s in WATCH}).sort_index())

print("\nLibrary correlation (datewise avg spearman rho):")
print("  flip_mom vs mom_diff:", round(spearman_panel_rho(f_flip, f_momd), 4))

# Macro reference: VIX level regime
mac = load_macro(ASOF)
if 'VIX' in mac.columns and len(mac['VIX'].dropna()):
    v = mac['VIX'].dropna()
    from zscore_series_helper import dummy  # no-op
print("\nGATE: |IC|>=0.0070, |ICIR|>=0.0840")
print("DEPRECATE rule: FAIL iff recent(30/60/180d) IC sign flips or |ICIR| clearly negative at full sample")