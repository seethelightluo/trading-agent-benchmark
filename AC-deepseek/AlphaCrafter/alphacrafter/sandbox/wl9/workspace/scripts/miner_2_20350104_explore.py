"""
miner_3 2034-12-07: Explore novel cross-asset factor candidates.
Universe: 15 tradable cross-asset instruments.
Data window: 2020-01 .. 2034-12-06 (visible-through). 
Gates: abs(paper IC)>=0.0070 AND abs(paper ICIR)>=0.0840 at 10d horizon, min 8 assets/date.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2035-01-03'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets,end):
    closes={}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date'])
        df=df[df['date']<=pd.Timestamp(end)].sort_values('date').set_index('date')
        closes[a]=df['close'].astype(float)
    return pd.DataFrame(closes)

close=load(ASSETS,VISIBLE_END).dropna(how='all')
close=close.dropna(axis=1,how='all')
print(f"Panel {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}",flush=True)
rets=close.pct_change()

def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5,fwd10,fwd20=fwd(5),fwd(10),fwd(20)

def compute_ic(fv,fw,min_dates=30,start=None,flip=False):
    if flip: fv=-fv
    f=fv.reindex(fw.index); ii=fw.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[];ok=0
    for d in ii:
        x=f.loc[d]; y=fw.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean();sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),
                hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)

def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name,fv,start=None,flip=True):
    a=compute_ic(fv,fwd10,start=start,flip=flip)
    b=compute_ic(fv,fwd5,start=start,flip=flip); c=compute_ic(fv,fwd20,start=start,flip=flip)
    ok=(abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084)
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok_d={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}",flush=True)
    return a,ok
def compute_metrics(fv,fw,min_dates=30,start=None,flip=False):
    return compute_ic(fv,fw,min_dates,start,flip)

def load_hilo(assets,end):
    H,L,V={},{},{}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=pd.Timestamp(end)].sort_values('date').set_index('date')
        H[a]=df['high'].astype(float); L[a]=df['low'].astype(float)
        V[a]=df['volume'].astype(float) if ('volume' in df and df['volume'].notna().any()) else pd.Series(np.nan,index=df.index)
    return pd.DataFrame(H).reindex(close.index), pd.DataFrame(L).reindex(close.index), pd.DataFrame(V).reindex(close.index)
high,low,vol=load_hilo(ASSETS,VISIBLE_END)

# C1: 60d momentum skip5
c1 = close/close.shift(65)-1
# C2: downside semi-deviation 20d
ds = rets.clip(upper=0)
c2 = ((ds**2).rolling(20).mean()).apply(np.sqrt)
# C3: 10d-avg range-position (close within high/low)
c3 = ((close-low)/(high-low+1e-12)).rolling(10).mean()

RUN='2032-01-01'
print('=== MULTI-SAMPLE (2021+) ===',flush=True)
for nm,fv,fl in [('mom60_skip5',c1,False),('downside_sd_20',c2,False),('rng_pos_10',c3,False),
                 ('mom60_skip5_contra',c1,True),('downside_sd_20_contra',c2,True),('rng_pos_10_contra',c3,True)]:
    report(nm,fv,start='2021-01-01',flip=fl)

print('=== RECENT (2032+) ===', flush=True)
for nm,fv in [('mom60_skip5',c1),('downside_sd_20',c2),('rng_pos_10',c3)]:
    report(nm,fv,start=RUN,flip=False)
    report(nm+'_contra',fv,start=RUN,flip=True)
