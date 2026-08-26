"""miner_3 2034-11-09 cycle: fresh candidate factor exploration on the 15-instrument
cross-asset tradable universe. Data as-of visible_through (2034-11-08) from date.json.
No lookahead. Admission gates (shared, 10d paper horizon): |IC| >= 0.0070,
|ICIR| >= 0.0840. Candidates focus on cross-asset rotation/mean-reversion themes.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)}", flush=True)

def vseries(s): return s.dropna()
def forward(s, h):
    v = vseries(s); return (v.shift(-h) / v - 1.0).reindex(INDEX)
def build(df): return df.sort_index().replace([np.inf, -np.inf], np.nan).astype(float)
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

fwd10 = pd.DataFrame({s: forward(px[s], 10) for s in WATCH}).sort_index()

def f_riskadj_mom(df, l=60, s=20):
    c = df['close']
    mom = c.pct_change(l)
    r = c.pct_change()
    vol = r.rolling(s).std().replace(0, np.nan)
    return mom / vol

def f_drawdown_dist(df, l=60):
    c = df['close']
    rollmax = c.rolling(l, min_periods=10).max()
    return (c - rollmax) / rollmax.replace(0, np.nan)

def f_vol_term(df, s=10, l=60):
    c = df['close']; r = c.pct_change()
    vs = r.rolling(s).std(); vl = r.rolling(l).std()
    return vs / vl.replace(0, np.nan)

def f_upcapture(df, s=20, l=60):
    c = df['close']; r = c.pct_change()
    up = r.where(r > 0).rolling(s).mean()
    dn = (-r).where(r < 0).rolling(s).mean()
    return up / dn.replace(0, np.nan)

def f_range_compress(df, s=5, l=60):
    c = df['close']; r = c.pct_change()
    vs = r.rolling(s).std(); vl = r.rolling(l).std()
    return vs / vl.replace(0, np.nan)

def group_spread(up_list, low_list, s=20):
    up = pd.concat([px[x]['close'].pct_change(s).rename(x) for x in up_list], axis=1).mean(axis=1)
    low = pd.concat([px[x]['close'].pct_change(s).rename(x) for x in low_list], axis=1).mean(axis=1)
    spread = (up - low).reindex(INDEX)
    out = pd.DataFrame(index=INDEX)
    for sym in WATCH:
        out[sym] = px[sym]['close'].pct_change(s) - spread
    return out

EQ = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E', 'NDX', 'SOX']
CRYPTO = ['BTC', 'ETH']
COMM = ['XAU', 'COPPER', 'WTI']

print("=== candidates (10d fwd) ===", flush=True)
cands = {}
cands['R1_range_compress_5x60'] = zcross(panel(lambda df: f_range_compress(df, 5, 60)))
cands['R2_upcapture_20x60'] = zcross(panel(f_upcapture))
cands['R3_vol_term_10x60'] = zcross(panel(f_vol_term))
cands['R4_drawdown_dist_60'] = zcross(panel(f_drawdown_dist))
cands['R5_crypto_xeq_mom20'] = zcross(group_spread(CRYPTO, EQ, 20))
cands['R6_riskadj_mom60x20'] = zcross(panel(f_riskadj_mom))
cands['R7_comm_vs_xeq_mom10'] = zcross(group_spread(COMM, EQ, 10))
cands['R8_crypto_vs_comm_mom10'] = zcross(group_spread(CRYPTO, COMM, 10))

for lab, fac in cands.items():
    ic_report(fac, fwd10, lab)

print("=== decay (5/15/20d) ===", flush=True)
for lab, fac in cands.items():
    for h in (5, 15, 20):
        fh = pd.DataFrame({s: forward(px[s], h) for s in WATCH}).sort_index()
        icd = cross_sectional_ic(fac, fh)
        st = ic_stats(icd)
        print(f"  {lab} h={h}d: IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} n={st['n_dates']}", flush=True)

print("=== library correlation ===", flush=True)
lib_paths = sorted(glob.glob('factors/*.json'))
lib_path
lib_paths = [p for p in lib_paths if 'bak' not in p and 'evicted' not in p and 'rejected' not in p and 'ensemble' not in p and 'deprecated' not in p]
print("library files:", len(lib_paths), flush=True)
# Self-check correlation vs library proxies recomputable from price panel (mom_10d_skip5 as the
# canonical library proxy). Exact rho against all library signal artifacts is the post-Miner gate.
mom10 = zcross(panel(lambda df: df['close'].pct_change(5) / df['close'].shift(15) - 1.0))
for lab, fac in cands.items():
    r = spearman_panel_rho(fac, mom10)
    print(f"  {lab}: rho_vs_mom10_proxy={r:+.3f}", flush=True)
print("done", flush=True)
