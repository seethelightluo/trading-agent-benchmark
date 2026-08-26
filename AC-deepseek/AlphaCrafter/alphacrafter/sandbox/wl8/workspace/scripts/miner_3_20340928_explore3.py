"""miner_3 2034-09-28 cycle: fresh candidate factor exploration on the 15-instrument
cross-asset tradable universe. Data as-of visible_through from ../persistent/date.json.
No lookahead. Gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 on 10d paper horizon.
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
    vol = r.rolling(s).std()
    return mom / vol

def f_downslope(df, s=20, l=60):
    c = df['close']; r = c.pct_change()
    dv = r.where(r < 0).rolling(s).std()
    uv = r.where(r > 0).rolling(s).std()
    ratio = dv / uv.replace(0, np.nan)
    return ratio - ratio.rolling(l).mean()

def f_escape(df, s=20, l=60):
    c = df['close']
    rng = (c.rolling(s).max() - c.rolling(s).min()).replace(0, np.nan)
    pos = (c - c.rolling(s).min()) / rng
    return pos - pos.rolling(l).mean()

def group_spread(up_list, low_list, s=20):
    up = pd.concat([px[x]['close'].pct_change(s).rename(x) for x in up_list], axis=1).mean(axis=1)
    low = pd.concat([px[x]['close'].pct_change(s).rename(x) for x in low_list], axis=1).mean(axis=1)
    spread = (up - low).reindex(INDEX)
    out = pd.DataFrame(index=INDEX)
    for sym in WATCH:
        out[sym] = px[sym]['close'].pct_change(s) - spread
    return out

print("=== candidates ===", flush=True)

f1 = group_spread(['BTC', 'ETH'], ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E', 'NDX', 'SOX'], 20)
ic_report(zcross(f1), fwd10, 'C1_crypto_xeq_mom20')

f2 = group_spread(['NDX', 'SOX'], ['XAU'], 20)
ic_report(zcross(f2), fwd10, 'C2_tech_vs_gold_mom20')

f3 = panel(f_riskadj_mom)
ic_report(zcross(f3), fwd10, 'C3_riskadj_mom60x20')

f4 = panel(f_downslope)
ic_report(zcross(f4), fwd10, 'C4_downslope_20x60')

f5 = panel(f_escape)
ic_report(zcross(f5), fwd10, 'C5_escape_20x60')

f6 = group_spread(['BTC', 'ETH'], ['XAU', 'COPPER', 'WTI'], 20)
ic_report(zcross(f6), fwd10, 'C6_crypto_vs_comm_mom20')

yc = (px['US10Y']['close'].pct_change(20) - px['CN10Y']['close'].pct_change(20)).reindex(INDEX)
f7 = pd.DataFrame(index=INDEX)
for sym in WATCH:
    f7[sym] = yc
ic_report(zcross(f7), fwd10, 'C7_yield_gap_mom20')

f8 = panel(lambda df: f_riskadj_mom(df, l=60, s=60))
ic_report(zcross(f8), fwd10, 'C8_riskadj_mom60x60')

print("=== decay ===", flush=True)
def decay_report(fac, label):
    for h in (5, 15, 20):
        fh = pd.DataFrame({s: forward(px[s], h) for s in WATCH}).sort_index()
        icd = cross_sectional_ic(fac, fh)
        s