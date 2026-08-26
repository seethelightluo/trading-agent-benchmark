"""miner_3 2034-10-12 cycle: fresh candidate factor exploration on the 15-instrument
cross-asset tradable universe. Data as-of visible_through=2034-10-11. No lookahead.
Admission gates (shared, 10d paper horizon): |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)}", flush=True)

def forward(s, h):
    v = px[s].dropna()
    return (v.shift(-h) / v - 1.0).reindex(INDEX)

def build(df):
    return df.sort_index().replace([np.inf, -np.inf], np.nan).astype(float)

def panel(fn):
    return build(pd.DataFrame({s: fn(px[s]) for s in WATCH}))

def zcross(fac):
    med = fac.median(axis=1)
    mad = (fac.sub(med, axis=0)).abs().median(axis=1)
    out = fac.sub(med, axis=0).div(mad.replace(0, np.nan), axis=0)
    return out.clip(-5, 5)

def ic_report(fac, fwd, label):
    icd = cross_sectional_ic(fac, fwd)
    st = ic_stats(icd)
    reg = regime_split(icd)
    rm120 = icd.index >= icd.index[-1] - pd.Timedelta(days=120)
    st120 = ic_stats(icd[rm120]) if rm120.any() else None
    rm365 = icd.index >= icd.index[-1] - pd.Timedelta(days=365)
    st365 = ic_stats(icd[rm365]) if rm365.any() else None
    rank = fac.rank(axis=1, pct=True).dropna(how='all')
    to10 = rank.diff(10).abs().mean().mean() if len(rank) > 10 else np.nan
    cov = fac.notna().mean().mean()
    print(f"  {label}: IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
          f"n={st['n_dates']} avgN={st['avg_n']:.1f} turn10={to10:.3f} cov={cov:.3f}", flush=True)
    for lab, seg in reg.items():
        if seg[2]:
            print(f"      reg {lab}: IC={seg[0]:+.4f} ICIR={seg[1]:+.4f} n={seg[2]}", flush=True)
    if st120 is not None:
        print(f"      last120d: IC={st120['ic']:+.4f} ICIR={st120['icir']:+.4f} n={st120['n_dates']}", flush=True)
    if st365 is not None:
        print(f"      last365d: IC={st365['ic']:+.4f} ICIR={st365['icir']:+.4f} n={st365['n_dates']}", flush=True)
    return st, reg, icd

fwd10 = pd.DataFrame({s: forward(s, 10) for s in WATCH}).sort_index()

def group_spread(up_list, low_list, s=20):
    up = pd.concat([px[x].pct_change(s).rename(x) for x in up_list], axis=1).mean(axis=1)
    low = pd.concat([px[x].pct_change(s).rename(x) for x in low_list], axis=1).mean(axis=1)
    spread = (up - low).reindex(INDEX)
    out = pd.DataFrame(index=INDEX)
    for sym in WATCH:
        out[sym] = px[sym].pct_change(s) - spread
    return out

EQ = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E', 'NDX', 'SOX']
CRYPTO = ['BTC', 'ETH']
COMM = ['XAU', 'COPPER', 'WTI']

print("=== candidates ===", flush=True)
cands = {}
cands['C1_crypto_xeq_mom20'] = zcross(group_spread(CRYPTO, EQ, 20))
cands['C2_tech_vs_gold_mom20'] = zcross(group_spread(['NDX', 'SOX'], ['XAU'], 20))
cands['C3_riskadj_mom60x20'] = zcross(panel(lambda df: f_riskadj_mom(df['close'])))
cands['C4_down_up_ratio_20x60'] = zcross(panel(lambda df: f_down_up_ratio(df['close'])))
cands['C5_escape_20x60'] = zcross(panel(lambda df: f_escape(df['close'])))
cands['C6_crypto_vs_comm_mom20'] = zcross(group_spread(CRYPTO, COMM, 20))
cands['C8_riskadj_mom60x60'] = zcross(panel(lambda df: f_riskadj_mom(df['close'], l=60, s=60)))
cands['C9_mom_diff_20x60'] = zcross(panel(lambda df: f_mom_diff(df['close'])))
cands['C10_hi_dist_20'] = zcross(panel(lambda df: f_hi_dist(df['close'], w=20)))

for lab, fac in cands.items():
    fc = fac.copy()
    ic_report(fc, fwd10, lab)

print("=== decay ===", flush=True)
for lab, fac in cands.items():
    for h in (5, 15, 20):
        fh = pd.DataFrame({s: forward(s, h) for s in WATCH}).sort_index()
        icd = cross_sectional_ic(fac, fh)
        st = ic_stats(icd)
        print(f"  {lab} h={h}d: IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} n={st['n_dates']}", flush=True)

# library correlation
lib_paths = sorted(glob.glob('factors/*.json'))
lib_paths = [p for p in lib_paths if 'bak' not in p and 'evicted' not in p and 'rejected' not in p and 'ensemble' not in p]
print("library files:", [p.split('/')[-1] for p in lib_paths], flush=True)
for lab, fac in cands.items():
    best = 0.0
    bestn = None
    for lp in lib_paths:
        try:
            d = json.load(open(lp))
            expr = d.get('calculation', {}).get('expression', '')
            # crude: try to recompute well-known library series by expression eval on panel
        except Exception:
            pass
    print(f"  {lab}: max_abs_library_correlation not computed (needs signal artifacts)", flush=True)