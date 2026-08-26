"""miner_3 2033-11-24: full-sample validation of best fresh candidates + decay + multi-regime.
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
r=close.pct_change().dropna()
rng=(high-low)/close

def compute_ic(fv,fwd,start=None,flip=False):
    f=fv.reindex(fwd.index)
    if flip: f=-f
    ics=[]; ok=0
    for d in fwd.index:
        if start and d<pd.Timestamp(start): continue
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<30: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),hit=float((ics>0).mean()))

def fwd(h): return r.shift(-h).rolling(h).mean()
f5=fwd(5); f10=fwd(10); f20=fwd(20)
def report(name,fv,flip=False):
    full=compute_ic(fv,f10,flip=flip)
    rec=compute_ic(fv,f10,start='2032-01-01',flip=flip)
    f5v=compute_ic(fv,f5,start='2032-01-01',flip=flip)['IC']
    f20v=compute_ic(fv,f20,start='2032-01-01',flip=flip)['IC']
    print(f"{name}: full IC={full['IC']:.4f} ICIR={full['ICIR']:.2f} n={full['n']} hit={full['hit']:.3f} | rec IC={rec['IC']:.4f} ICIR={rec['ICIR']:.2f} | f5/f20 rec={f5v:.3f}/{f20v:.3f}")
    return full,rec

print("FULL SAMPLE + DECAY (flip=optimal direction from sweep):")
report('range_ratio_20_60', rng.rolling(20).mean()/rng.rolling(60).mean()-1.0, flip=True)
report('kaufman_eff_60', (close.diff(60).abs()).div(close.diff().abs().rolling(60).sum()), flip=False)
report('zscore_20',(close-close.rolling(20).mean())/close.rolling(20).std().replace(0,np.nan), flip=True)
report('below_high60', close/close.rolling(60).max()-1.0, flip=True)
# semi-vol and vol_ratio for comparison
down=r.clip(upper=0)
report('semi_vol_20', down.rolling(20).std(), flip=True)
report('vol_ratio_60_10', r.rolling(60).std()/r.rolling(10).std().replace(0,np.nan), flip=True)