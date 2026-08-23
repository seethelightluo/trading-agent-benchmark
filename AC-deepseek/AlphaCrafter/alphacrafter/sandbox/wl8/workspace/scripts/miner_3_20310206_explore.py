"""miner_3 2031-02-06: explore new candidate factors for the 15-asset cross-asset benchmark.
Test several interpretable factor ideas over full history + recent regimes, decay, coverage.
Admission gate (benchmark): |IC|>=0.0070, |ICIR|>=0.0840 on full history.
"""
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
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s)
    return (v.shift(-h)/v - 1.0).reindex(INDEX)
def rollz(s, win):
    v = vseries(s)
    r = v.pct_change()
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
    line = (f"{name:24s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg_n={st.get('avg_n',np.nan):4.1f} cov={cov:.3f} | "
            f"365d IC={ic365['ic']:+.4f}/{ic365['icir']:+.4f} last60 IC={ic60['ic']:+.4f}/{ic60['icir']:+.4f}")
    print(line)
    for lab, seg in regime_split(icd).items():
        print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st

cands = {}

# 1. Risk-adjusted momentum 20d: mom20 / realized vol20
v20 = pd.DataFrame({s: -vseries(px[s]).pct_change().rolling(20).std() for s in WATCH}).sort_index()
mom20 = build(pd.DataFrame({s: retk(px[s], 20) for s in WATCH}))
cands['risk_adj_mom_20'] = (mom20 + v20)

# 2. Vol state 20 (low-vol tilt): negative realized vol already in #1; test raw low-vol
cands['vol_low_20'] = v20  # negative 20d realized vol

# 3. Range position 20d
cands['range_pos_20'] = build(pd.DataFrame({s: (vseries(px[s]) - vseries(px[s]).rolling(20).min()) /
                                            (vseries(px[s]).rolling(20).max() - vseries(px[s]).rolling(20).min())
                                            for s in WATCH}))

# 4. Drawdown 20 (close vs rolling max)
cands['drawdown_20'] = build(pd.DataFrame({s: (vseries(px[s])/vseries(px[s]).rolling(20).max()-1.0) for s in WATCH}))

# 5. Trend ft: (close - sma20)/close
cands['trend_sma20'] = build(pd.DataFrame({s: (vseries(px[s])-vseries(px[s]).rolling(20).mean())/vseries(px[s]).rolling(20).mean() for s in WATCH}))

# 6. MACD-style: mom20 - mom60
cands['mom_diff_20_60'] = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))

# 7. Gold-tilt / defensive-bid: cross-asset relative strength of XAU vs SPX (only meaningful per-date, skip)

for name, fd in cands.items():
    d = fd.copy()
    assess(name, d)

print("\nGATE: |IC|>=0.0070 |ICIR|>=0.0840 (full history)")