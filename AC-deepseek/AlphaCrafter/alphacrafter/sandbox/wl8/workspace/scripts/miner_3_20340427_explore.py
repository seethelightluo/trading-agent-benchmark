"""miner_3 2034-04-27 cycle: fresh candidate factor exploration on the 15-instrument
cross-asset tradable universe. Data as-of visible_through from ../persistent/date.json.
No lookahead: factor at t uses data <= t; forward return t..t+h computed on close prices.

Admission gates (shared with other miners, 15-instrument universe):
  |IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d paper horizon.
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

def vseries(s): return s.dropna()
def retk(s, k):
    return (vseries(s) / vseries(s).shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h) / v - 1.0).reindex(INDEX)
def build(df): return df.sort_index().replace([np.inf, -np.inf], np.nan).astype(float)

def panel(fn):
    return build(pd.DataFrame({s: fn(px[s]) for s in WATCH}))

def ic_report(fac, fwd, label):
    icd = cross_sectional_ic(fac, fwd)
    st = ic_stats(icd)
    # regime
    reg = regime_split(icd)
    # recency
    rm120 = icd.index >= icd.index[-1] - pd.Timedelta(days=120)
    st120 = ic_stats(icd[rm120]) if rm120.any() else None
    rm365 = icd.index >= icd.index[-1] - pd.Timedelta(days=365)
    st365 = ic_stats(icd[rm365]) if rm365.any() else None
    # turnover of ranks (10d diff)
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
    return st, reg

# forward returns @10d
fwd10 = pd.DataFrame({s: forward(px[s], 10) for s in WATCH}).sort_index()

# ---------------- candidate factor definitions ----------------
def f_vol_target_mom(df, s=20, l=60):
    c = df['close'].values
    mom = np.full(len(c), np.nan)
    vol = np.full(len(c), np.nan)
    if len(c) > l:
        mom[l:] = c[l:] / c[:-l] - 1
    if len(c) > s:
        r = np.diff(c) / c[:-1]
        vol[s:] = pd.Series(r).rolling(s).std().values
    return pd.Series(mom / vol, index=df.index)

def f_sma_dist(df, w=40):
    c = df['close']
    return c / c.rolling(w).mean() - 1

def f_hi_dist(df, w=20):
    c = df['close']
    return c / c.rolling(w).max() - 1

def f_lo_dist(df, w=20):
    c = df['close']
    return c / c.rolling(w).min() - 1

def f_drawdown_recovery(df, w=40):
    c = df['close']
    return 1 - c / c.rolling(w).max()

def f_gap_intensity(df, w=20):
    df = df.copy()
    o = df['open']; pc = df['close'].shift(1)
    g = (o / pc - 1).abs()
    return g.rolling(w).mean()

def f_close_loc(df, w=20):
    df = df.copy()
    hl = (df['high'] - df['low']).replace(0, np.nan)
    cl = (df['close'] - df['low']) / hl
    return cl.rolling(w).mean()

def f_down_freq(df, w=60):
    r = df['close'].pct_change()
    return (r < 0).rolling(w).mean()

def f_updown_asym(df, w=20):
    r = df['close'].pct_change()
    up = r.where(r > 0); dn = r.where(r < 0)
    upm = up.rolling(w).mean(); dnm = dn.rolling(w).mean()
    return upm - dnm

def f_gk_vol_ratio(df, s=10, l=40):
    df = df.copy()
    o, h, l_, c = df['open'], df['high'], df['low'],