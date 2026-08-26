\"""miner_3 cycle 2033-04-14. Visible through 2033-04-13. No lookahead.
Revalidate effective library, sweep new candidates. Gates: abs IC>=0.0070, abs ICIR>=0.084 (10d).
Small 15-asset cross-asset universe; report dates/instruments used explicitly."""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-04-13'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(assets,end):
    C,H,L,V={},{},{},{}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=end].sort_values('date').set_index('date')
        C[a]=df['close'].astype(float); H[a]=df['high'].astype(float); L[a]=df['low'].astype(float)
        V[a]=df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan,index=df.index)
    return C,H,L,V
closes,highs,lows,vols=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(closes).dropna(); high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index)
vol=pd.DataFrame(vols).reindex(close.index)
rets=close.pct_change().dropna()
idx=close.index
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}",flush=True)
def mac(c):
    df=pd.read_csv(ID/f'{c}.csv',parse_dates=['date']); return df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float).reindex(idx)
vix=mac('VIX'); usdcny=mac('USDCNY'); dxyr=mac('DXY').pct_change()
dVIX=vix.pct_change(); dCNY=usdcny.pct_change()

def roll_acorr(x, win):
    out=pd.Series(np.nan,index=x.index); xc=x.values; n=len(xc)
    for i in range(win-1,n):
        seg=xc[i-win+1:i+1]
        if np.std(seg)>0 and len(seg)>3: out.iloc[i]=np.corrcoef(seg[1:],seg[:-1])[0,1]
    return out
ac1=pd.DataFrame({c: roll_acorr(close[c],120) for c in close.columns})

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
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok,tov=0.)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),
                hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok,tov=0.)
def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0); return float((s.diff()!=0).mean().mean())
def report(name,fv,start=None,flip=False):
    f=fv.reindex(fwd10.index)
    a=compute_ic(f,fwd10,start=start,flip=flip)
    b=compute_ic(f,fwd5,start=start,flip=flip); c=compute_ic(f,fwd20,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.0