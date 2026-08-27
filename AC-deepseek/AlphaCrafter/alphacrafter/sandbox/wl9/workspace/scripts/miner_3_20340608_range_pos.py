"""
miner_3 2034-06-08: Explore novel factor 'year_range_252'
Position of current close within trailing [low, high] range over 252d.
Hypothesis: cross-asset 'range position' quality/reversal signal.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2034-06-07'
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
close=pd.DataFrame(closes); high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index)
close=close.dropna(how='all'); high=high.reindex(close.index); low=low.reindex(close.index)
rets=close.pct_change().dropna()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}",flush=True)
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
def compute_ic(fv,fw,min_dates=30,start=None,flip=False):
    f=fv.reindex(fw.index)
    if flip: f=-f
    ii=fw.index
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
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0); return float((s.diff()!=0).mean().mean())
def report(name,fv,start=None,flip=True):
    a=compute_ic(fv,fwd10,start=start,flip=flip)
    b=compute_ic(fv,fwd5,start=start,flip=flip); c=compute_ic(fv,fwd20,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok_dates={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}",flush=True)
    return a,ok
def compute_metrics(fv,fw,min_dates=30,start=None,flip=False):
    return compute_ic(fv,fw,min_dates,start,flip)
RUN='2032-01-01'
yr=(close - low.rolling(252).min())/(high.rolling(252).max()-low.rolling(252).min())
print("=== year_range_252 FULL (2022+) ===",flush=True)
a,ok=report('year_range_252 flip1(contra)',yr,start='2022-01-01',flip=True)
print("=== year_range_252 RECENT (2032+) ===",flush=True)
report('year_range_252 flip1(contra)',yr,start='2032-01-01',flip=True)
report('year_range_252 flip0(pro) ',yr,start='2032-01-01',flip=False)