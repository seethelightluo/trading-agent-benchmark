"""miner_3 2033-10-13. Full-period + recent stability for top low-corr candidates; expand corr lib."""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-10-12'
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
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
def compute_ic(fv,fwd,start=None,flip=False):
    f=fv.reindex(fwd.index)
    if flip: f=-f
    ii=fwd.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[]
    for d in ii:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8 and np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
            ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics); mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),hit=float((ics>0).mean()))
def report(name,fv,start=None,flip=False):
    a=compute_ic(fv,fwd10,start=start,flip=flip)
    b=compute_ic(fv,fwd5,start=start,flip=flip); c=compute_ic(fv,fwd20,start=start,flip=flip)
    print(f"{name} full[10]{a['IC']:.4f}/{a['ICIR']:.3f} n={a['n']} |[5]{b['IC']:.3f}[20]{c['IC']:.3f}",flush=True)

R='2031-06-15'
print("=== FULL PERIOD (since 2020) === ")
report("retrace_120",close/close.rolling(120).max()-1.0,flip=True)
report("retrace_60",close/close.rolling(60).max()-1.0,flip=True)
report("mom_accel_10x60",(close.shift(5)/close.shift(15)-1.0)-(close.shift(5)/close.shift(65)-1.0))
report("mom_240",close.shift(5)/close.shift(245)-1.0,flip=True)
print("\n=== RECENT (2021-06-15) ===")
report("retrace_120",close/close.rolling(120).max()-1.0,start=RAND,flip=True)
report("retrace_60",close/close.rolling(60).max()-1.0,start=RAND,flip=True)
report("mom_accel_10x60",(close.shift(5)/close.shift(15)-1.0)-(close.shift(5)/close.shift(65)-1.0),start=RAND)
report("mom_240",close.shift(5)/close.shift(245)-1.0,start=RAND,flip=True)