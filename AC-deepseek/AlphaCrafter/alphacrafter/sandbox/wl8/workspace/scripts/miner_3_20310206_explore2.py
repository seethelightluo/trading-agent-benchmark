"""miner_3 2031-02-06 explore batch 2: additional candidate factors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split)

ASOF = load_visible_through()
px = load_prices(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s); return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v - 1.0).reindex(INDEX)
def rollz(s, win):
    v = vseries(s); r = v.pct_change()
    m = r.rolling(win).mean(); sd = r.rolling(win).std()
    return ((r - m)/sd).reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan)

def assess(name, factor_df):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    ic60 = ic_stats(icd.tail(60))
    ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)] if len(icd) else icd)
    line = (f"{name:26s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} cov={cov:.3f} | "
            f"365d IC={ic365['ic']:+.4f}/{ic365['icir']:+.4f} last60 IC={ic60['ic']:+.4f}/{ic60['icir']:+.4f}")
    print(line)
    for lab, seg in regime_split(icd).items():
        print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    # gate
    gate = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    print(f"    GATE: {'PASS' if gate else 'FAIL'}")
    return st

cands = {}

# A. Relative momentum vs cross-sectional median 10d
mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
med10 = mom10.median(axis=1)
cands['rel_mom10_vs_med'] = (mom10.sub(med10, axis=0))

# B. XAU-relative strength vs equity/commodity basket (defensive momentum)
# XAU vs SPX spread momentum over 20d per-symbol macro-relative (not cross-sectional) - skip

# C. Crypto momentum carry indicator: BTC+ETH avg 20d - cross asset median
crypto = build(pd.DataFrame({s: retk(px[s],20) for s in ['BTC','ETH']}))
cand_crypto_all = build(pd.DataFrame({s: retk(px[s],20) for s in WATCH}))
crypto_mean = crypto.mean(axis=1)
cands['crypto_mom20_spread'] = cand_crypto_all.sub(crypto_mean, axis=0)

# D. Commodity momentum spread: WTI+COPPER+XAU avg 20d minus all-mean
comm_all = build(pd.DataFrame({s: retk(px[s],20) for s in WATCH}))
comm_avg = build(pd.DataFrame({s: retk(px[s],20) for s in ['WTI','COPPER','XAU']})).mean(axis=1)
cands['commodity_mom20_spread'] = comm_all.sub(comm_avg, axis=0)

# E. 20d mean-reversion: negative short momentum (fade recent winner)
cands['mom_rev_20'] = build(pd.DataFrame({s: -retk(px[s],20) for s in WATCH}))
cands['mom_rev_10'] = build(pd.DataFrame({s: -retk(px[s],10) for s in WATCH}))

# F. Vol-of-vol 20x60 asymmetry (realized vol rising)
def rv(s, win=20):
    v = vseries(s); return v.pct_change().rolling(win).std()
vol20 = pd.DataFrame({s: rv(px[s]) for s in WATCH})
vol60 = pd.DataFrame({s: rv(px[s],60) for s in WATCH})
cands['vol_ratio_20_60'] = build(vol20/vol60)

# G. Cross-section z-score of 20d return (dispersion-adjusted momentum)
def csz(df):
    out = pd.DataFrame(index=df.index, columns=df.columns)
    for d in df.index:
        row = df.loc[d]
        good = row.dropna()
        if len(good)>=6:
            mu = good.mean(); sd = good.std()
            if sd and not np.isnan(sd) and sd>0:
                out.loc[d] = (row-mu)/sd
    return out
cands['zscore_mom20'] = build(csz(mom10))

for name, fd in cands.items():
    assess(name, fd)
print("\nDONE")