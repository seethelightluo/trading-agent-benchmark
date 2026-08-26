"""miner_2 2035-09-13 cycle: novel candidate factor exploration on the 15-instrument
cross-asset tradable universe. Data as-of visible_through (2035-09-12) from date.json.
No lookahead: factors use data through t; forward returns t..t+h.
Admission gates (shared, 10d paper horizon): |IC| >= 0.0070, |ICIR| >= 0.0840.
Uses miner_3_20261203_common helpers for loading and IC stats.
"""
import sys, os, json
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
assert px.shape[1] == len(WATCH), "universe width mismatch"

def forward(s, h):
    v = s.dropna()
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
    gate = (abs(st['ic']) >= 0.0070 and abs(st['icir']) >= 0.0840)
    print(f"  {label}: IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
          f"n={st['n_dates']} avgN={st['avg_n']:.1f} turn10={to10:.3f} cov={cov:.3f} "
          f"GATE={'PASS' if gate else 'FAIL'}", flush=True)
    for lab, seg in reg.items():
        if seg[2]:
            print(f"      reg {lab}: IC={seg[0]:+.4f} ICIR={seg[1]:+.4f} n={seg[2]}", flush=True)
    if st120 is not None:
        print(f"      last120d: IC={st120['ic']:+.4f} ICIR={st120['icir']:+.4f} n={st120['n_dates']}", flush=True)
    if st365 is not None:
        print(f"      last365d: IC={st365['ic']:+.4f} ICIR={st365['icir']:+.4f} n={st365['n_dates']}", flush=True)
    return st, reg, icd

fwd10 = pd.DataFrame({s: forward(px[s], 10) for s in WATCH}).sort_index()

# ---------- candidate factor library ----------
def f_raw_mom(c, l=10):
    return c.pct_change(l)

def f_vol_adj_mom(c, l=80, s=20):
    mom = c.pct_change(l)
    r = c.pct_change()
    vol = r.rolling(s).std().replace(0, np.nan)
    return mom / vol

def f_drawdown_depth(c, l=180):
    rollmax = c.rolling(l, min_periods=20).max()
    return (c - rollmax) / rollmax.replace(0, np.nan)

def f_drawdown_recovery(c, l=120):
    rollmax = c.rolling(l, min_periods=20).max()
    rollmin = c.rolling(l, min_periods=20).min()
    rng = (rollmax - rollmin).replace(0, np.nan)
    return (c - rollmin) / rng

def f_vol_term(c, s=10, l=90):
    r = c.pct_change()
    vs = r.rolling(s).std(); vl = r.rolling(l).std()
    return vs / vl.replace(0, np.nan)

def f_vol_ratio_40x160(c, s=40, l=160):
    r = c.pct_change()
    vs = r.rolling(s).std(); vl = r.rolling(l).std()
    return vs / vl.replace(0, np.nan)

def f_hit_rate(c, s=30):
    return (c.pct_change() > 0).rolling(s).mean()

def f_updown_ratio(c, s=20):
    r = c.pct_change()
    up = r.where(r > 0).rolling(s).mean()
    dn = (-r).where(r < 0).rolling(s).mean()
    return up / dn.replace(0, np.nan)

def f_hl_position(c, s=60):
    rollmax = c.rolling(s).max(); rollmin = c.rolling(s).min()
    rng = (rollmax - rollmin).replace(0, np.nan)
    return (c - rollmin) / rng

def f_mix_mom(c, l=40, s=20):
    m = c.pct_change(l)
    m_prev = c.pct_change(l).shift(l // 2)
    r = c.pct_change(); vol = r.rolling(s).std().replace(0, np.nan)
    return (m - m_prev) / vol

def f_macro_beta_vix(c, s=60, l=30):
    # rolling beta of asset returns vs VIX return (negative = defensive), z-scored cross-sectionally later
    vix = mac['VIX'].reindex(c.index)
    ar = c.pct_change(); vr = vix.pct_change()
    cov = ar.rolling(s).cov(vr); var = vr.rolling(s).var().replace(0, np.nan)
    beta = cov / var
    return beta

def f_macro_beta_dxy(c, s=60):
    dxy = mac['DXY'].reindex(c.index)
    ar = c.pct_change(); dr = dxy.pct_change()
    cov = ar.rolling(s).cov(dr); var = dr.rolling(s).var().replace(0, np.nan)
    return cov / var

def f_yield_gap(c, s=20):
    # assets vs a 10y yield series spread (bond-like spreads); use absolute level diff normalized
    us10 = px['US10Y']
    # approximate: asset momentum minus yield momentum
    return c.pct_change(s) - us.reindex(c.index).pct_change(s)

def group_spread(hot_list, cold_list, s=20):
    hot = pd.concat([px[x].pct_change(s).rename(x) for x in hot_list], axis=1).mean(axis=1)
    cold = pd.concat([px[x].pct_change(s).rename(x) for x in cold_list], axis=1).mean(axis=1)
    spread = (hot - cold).reindex(INDEX)
    out = pd.DataFrame(index=INDEX)
    for sym in WATCH:
        out[sym] = px[sym].pct_change(s) - spread
    return out

EQ = ['000300.SH','000688.SH','SPX','HSI','N225','SX