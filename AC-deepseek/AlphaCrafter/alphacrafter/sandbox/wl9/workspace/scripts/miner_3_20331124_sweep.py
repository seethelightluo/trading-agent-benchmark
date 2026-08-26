"""miner_3 cycle 2033-11-24. Visible through 2033-11-23. No lookahead.
Explore fresh candidate factors on recent regime 2032+.
Admission gates: abs daily paper IC>=0.0070, abs ICIR>=0.084 (10d fwd).
Cross-section >=8 names of the 15-instrument universe.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-11-23'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(assets,end):
    C,H,L={},{},{}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=end].sort_values('date').set_index('date')
        C[a]=df['close'].astype(float); H[a]=df['high'].astype(float); L[a]=df['low'].astype(float)
    return C,H,L
closes,highs,lows=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(closes).dropna(); high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index)
rets=close.pct_change().dropna()
ndx=close.index
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {ndx[0]:%Y-%m-%d}..{ndx[-1]:%Y-%m-%d}",flush=True)

def compute_ic(fv,fwd,min_dates=30,start=None,flip=False):
    f=fv.reindex(fwd.index)
    if flip: f=-f
    ii=fwd.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[]; ok=0
    for d in ii:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),
                hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)
def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0); return float((s.diff()!=0).mean().mean())
def report(name,fv,start=None,flip=False):
    a=compute_ic(fv,fwd10,start=start,flip=flip)
    b=compute_ic(fv,fwd5,start=start,flip=flip); c=compute_ic(fv,fwd20,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}",flush=True)
    return a
def rank_summaries(fv,flip=False):
    f=fv if not flip else -fv
    return f

RUN_START='2032-01-01'
print("=== FRESH CANDIDATES (recent 2032+) ===",flush=True)
rng=(high-low)/close
report("range_ratio_20_60",rng.rolling(20).mean()/rng.rolling(60).mean()-1.0,start=RUN_START,flip=True)
report("range_20",rng.rolling(20).mean(),start=RUN_START,flip=True)
report("upper_wave_20",((high-close)/close).rolling(20).mean(),start=RUN_START,flip=True)
report("lower_wave_20",((close-low)/close).rolling(20).mean(),start=RUN_START,flip=True)
hh=high.rolling(20).max(); ll=low.rolling(20).min()
stoch=(close-ll)/(hh-ll).replace(0,np.nan)
report("stoch_20",stoch,start=RUN_START,flip=True)
# distance from 20d mean / vol
report("zscore_20",(close-close.rolling(20).mean())/close.rolling(20).std().replace(0,np.nan),start=RUN_START,flip=True)
report("zscore_60",(close-close.rolling(60).mean())/close.rolling(60).std().replace(0,np.nan),start=RUN_START,flip=True)
# efficiency over 60d
report("kaufman_eff_60",(close.diff(60).abs()).div(close.diff().abs().rolling(60).sum()),start=RUN_START,flip=False)
# 5d momentum versus 20d reversal combo: short-term pullback after medium trend
m5=close.shift(1)/close.shift(6)-1.0
m20=close.shift(1)/close.shift(21)-1.0
report("mom_strength_5_20",m5-m20,start=RUN_START,flip=False)
# cross-sectional vol ratio (own vol vs short vol)
v60=rets.rolling(60).std(); v10=rets.rolling(10).std()
report("vol_ratio_60_10",v60/v10.replace(0,np.nan),start=RUN_START,flip=True)
# downside deviation (semi-vol) vs total vol 20d
down=rets.clip(upper=0)
report("semi_vol_20",down.rolling(20).std(),start=RUN_START,flip=True)
# mean reversion signal: distance below / above 20d low
report("below_high60",close/close.rolling(60).max()-1.0,start=RUN_START,flip=True)