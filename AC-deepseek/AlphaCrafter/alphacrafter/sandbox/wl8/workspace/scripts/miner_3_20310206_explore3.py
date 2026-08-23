"""miner_3 2031-02-06 explore batch 2 fixed: additional candidate factors."""
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

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def assess(name, factor_df):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)] if len(icd) else icd)
    ic60 = ic_stats(icd.tail(60))
    line = (f"{name:26s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} cov={cov:.3f} | "
            f"365d IC={ic365['ic']:+.4f}/{ic365['icir']:+.4f} last60 IC={ic60['ic']:+.4f}/{ic60['icir']:+.4f}")
    print(line)
    for lab, seg in regime_split(icd).items():
        print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    gate = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    print(f"    GATE: {'PASS' if gate else 'FAIL'}")
    return st

cands = {}

mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
med10 = mom10.median(axis=1)
cands['rel_mom10_vs_med'] = build(mom10.sub(med10, axis=0))

mom20 = build(pd.DataFrame({s: retk(px[s],20) for s in WATCH}))

crypto = build(pd.DataFrame({s: retk(px[s],20) for s in ['BTC','ETH']}))
crypto_mean = crypto.mean(axis=1)
cands['crypto_mom20_spread'] = build(mom20.sub(crypto_mean, axis=0))

comm_avg = build(pd.DataFrame({s: retk(px[s],20) for s in ['WTI','COPPER','XAU']})).mean(axis=1)
cands['commodity_mom20_spread'] = build(mom20.sub(comm_avg, axis=0))

cands['mom_rev_20'] = build(-mom20)
cands['mom_rev_10'] = build(-mom10)

def rv(s, win=20):
    v = vseries(s); return v.pct_change().rolling(win).std()
vol20 = pd.DataFrame({s: rv(px[s]) for s in WATCH})
vol60 = pd.DataFrame({s: rv(px[s],60) for s in WATCH})
cands['vol_ratio_20_60'] = build(vol20/vol60)

def csz(df):
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for d in df.index:
        row = df.loc[d].astype(float)
        good = row.dropna()
        if len(good)>=6:
            mu = good.mean(); sd = good.std()
            if sd and not np.isnan(sd) and sd>0:
                out.loc[d] = (row-mu)/sd
    return out
cands['zscore_mom10'] = build(csz(mom10))

for name, fd in cands.items():
    assess(name, fd)
print("\nDONE")