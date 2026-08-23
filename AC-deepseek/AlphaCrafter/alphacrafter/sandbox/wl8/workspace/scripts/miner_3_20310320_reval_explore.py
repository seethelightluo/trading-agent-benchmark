"""miner_3 2031-03-20: re-validate active factors + explore new candidates through visible_through."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s); return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v - 1.0).reindex(INDEX)
def rv(s, win):
    v = vseries(s); return v.pct_change().rolling(win).std().reindex(INDEX)
def mv(s, win):
    return vseries(s).rolling(win).mean().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def assess(name, factor_df):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    ic60 = ic_stats(icd.tail(60))
    ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)] if len(icd) else icd)
    gate = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    line = (f"{name:26s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg_n={st.get('avg_n',np.nan):4.1f} cov={cov:.3f} | "
            f"365d={ic365['ic']:+.4f}/{ic365['icir']:+.4f} last60={ic60['ic']:+.4f}/{ic60['icir']:+.4f} | {'PASS' if gate else 'FAIL'}")
    print(line)
    return st, gate, cov

print("\n===== ACTIVE FACTOR RE-VALIDATION =====")
# flip_mom_20x10
f_flip = build(pd.DataFrame({s: retk(px[s],20)*np.sign(retk(px[s],10)) for s in WATCH}))
assess("flip_mom_20x10", f_flip)
# mom_diff_20_60
f_momdd = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))
assess("mom_diff_20_60", f_momdd)
# usdcny_beta_60
f_usd = build(pd.DataFrame({s: retk(px[s],1).rolling(60).cov(retk(mac['USDCNY'],1))/retk(mac['USDCNY'],1).rolling(60).var() for s in WATCH}))
assess("usdcny_beta_60", f_usd)
# vol_low_20
vol20 = build(pd.DataFrame({s: rv(px[s],20) for s in WATCH}))
assess("vol_low_20", -vol20)

print("\n===== NEW CANDIDATES =====")
cands = {}
mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
mom20 = build(pd.DataFrame({s: retk(px[s],20) for s in WATCH}))
mom60 = build(pd.DataFrame({s: retk(px[s],60) for s in WATCH}))

# cross-sectional median momentum spread (relative momentum)
med20 = mom20.median(axis=1)
cands['rel_mom20'] = build(mom20.sub(med20, axis=0))
med10 = mom10.median(axis=1)
cands['rel_mom10'] = build(mom10.sub(med10, axis=0))
# negative drawdown (distance above rolling 20d high) -> strength
cands['range_pos_20'] = build(pd.DataFrame({s: (vseries(px[s])-vseries(px[s]).rolling(20).min())/(vseries(px[s]).rolling(20).max()-vseries(px[s]).rolling(20).min()) for s in WATCH}))
# drawdown from 20d max (higher = nearer high = stronger)
cands['dd_from_high_20'] = build(pd.DataFrame({s: (vseries(px[s])/vseries(px[s]).rolling(20).max()-1.0) for s in WATCH}))
# vol_ratio 20/60 (rising vs falling vol)
vol60 = build(pd.DataFrame({s: rv(px[s],60) for s in WATCH}))
cands['vol_ratio_20_60'] = build(vol20/vol60.replace(0,np.nan))
# trend: close vs sma20
cands['trend_sma20'] = build(pd.DataFrame({s: (vseries(px[s])-mv(px[s],20))/mv(px[s],20) for s in WATCH}))
# tendency: 5d vs 20d momentum diff (reversal timing)
cands['mom5_mom20'] = build(pd.DataFrame({s: retk(px[s],5)-retk(px[s],20) for s in WATCH}))
# RSI-14-like relative strength
def rsi(s, win=14):
    v = vseries(s); d = v.diff()
    up = d.clip(lower=0).rolling(win).mean(); dn = (-d.clip(upper=0)).rolling(win).mean()
    rs = up/(dn.replace(0,np.nan))
    return (100 - 100/(1+rs)).reindex(INDEX)
cands['rsi_14'] = build(pd.DataFrame({s: rsi(px[s]) for s in WATCH}))

for name, fd in cands.items():
    assess(name, fd)
print("\nDONE")
