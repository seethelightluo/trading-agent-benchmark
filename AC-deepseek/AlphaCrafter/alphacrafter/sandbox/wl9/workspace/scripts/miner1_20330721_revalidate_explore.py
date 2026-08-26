"""miner_1 cycle 2033-07-21. Visible through 2033-07-20. No lookahead.
Revalidate effective library in recent window + sweep fresh candidates.
Gates: abs IC>=0.0070, abs ICIR>=0.084 (10d), cross-section >=8 names.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-07-20'
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
C,H,L,V=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(C).dropna(); high=pd.DataFrame(H).reindex(close.index); low=pd.DataFrame(L).reindex(close.index)
rets=close.pct_change().dropna(); idx=close.index
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}",flush=True)
def mac(c):
    df=pd.read_csv(ID/f'{c}.csv',parse_dates=['date']); return df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float).reindex(idx)
vix=mac('VIX'); usdcny=mac('USDCNY'); usdjpy=mac('USDJPY'); dxy=mac('DXY')
dVIX=vix.pct_change(); dCNY=usdcny.pct_change(); dJPY=usdjpy.pct_change(); dDXY=dxy.pct_change()
def roll_acorr(x, win):
    out=pd.Series(np.nan,index=x.index); xc=x.values; n=len(xc)
    for i in range(win-1,n):
        seg=xc[i-win+1:i+1]
        if np.std(seg)>0 and len(seg)>3: out.iloc[i]=np.corrcoef(seg[1:],seg[:-1])[0,1]
    return out
ac1=pd.DataFrame({c: roll_acorr(close[c],120) for c in close.columns})
def beta_win(x,w): return rets.rolling(w).cov(x).div(x.rolling(w).var())
bv=beta_win(dVIX,60); cb=beta_win(dCNY,60); db=beta_win(dDXY,60)
ka=(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum())
mom120=close.shift(5)/close.shift(125)-1.0; mom10=close.shift(5)/close.shift(15)-1.0
bb=rets.rolling(20).std(); volz=rets.rolling(20).std().rank(axis=1)
skew=(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3))
kurt=(rets-rets.rolling(20).mean()).pow(4).rolling(20).mean().div(rets.rolling(20).std().pow(4))-3.0
# days since 60d high
dsh=pd.DataFrame(index=close.index,columns=close.columns,dtype=float)
for c in close.columns:
    v=close[c].values; out=np.zeros(len(v)); cur=0; rmax=close[c].rolling(60).max().values
    for i in range(60,len(v)):
        if v[i]>=rmax[i]-1e-12: cur=0
        else: cur+=1
        out[i]=cur
    dsh[c]=out
dsh=dsh.replace(0,np.nan) if False else dsh
# rng_pos_20d
rng_pos=pd.DataFrame(index=close.index,columns=close.columns,dtype=float)
for c in close.columns:
    hi=high[c].rolling(20).max(); lo=low[c].rolling(20).min()
    rng_pos[c]=(close[c]-lo)/(hi-lo)
# streak length 14
strk=pd.DataFrame(index=close.index,columns=close.columns,dtype=float)
for c in close.columns:
    diff=(close[c].diff()>0).astype(int).values; out=np.zeros(len(diff)); cur=0
    for i in range(len(diff)):
        cur=cur+1 if diff[i] else 0
        out[i]=cur
    strk[c]=out
# vix regret (mom10*vixsign)
vix_sign=np.sign(vix.diff(10).shift(5)).reindex(idx).ffill()
mom10_vixreg=(close/close.shift(5)-1.0).mul(vix_sign)
# vix_roc_20d (safe-haven factor)
vixroc20=vix.pct_change(20)
# helper to assign per-asset direction for vix_roc (library: long when roc up for havens, inverse risk)
havens=['XAU','US10Y','CN10Y']
def vixroc_factor():
    f=pd.DataFrame(index=idx,columns=close.columns,dtype=float)
    for c in close.columns:
        if c in havens: f[c]=vixroc20.reindex(idx)
        else: f[c]=-vixroc20.reindex(idx)
    return f
vixroc=vixroc_factor()

def compute_ic(fv,fwd,min_dates=30,start=None):
    ii=fwd.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[]
    for d in ii:
        x=fv.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),hit=float((ics>0).mean()))
def report(name,fv):
    f=fv.reindex(fwd10.index)
    a=compute_ic(f,fwd10); b=compute_ic(f,fwd5); c=compute_ic(f,fwd20)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name:26s} IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} n={a['n']:4d} hit={a['hit']:.3f} | [5]{b['IC']:+.3f}[20]{c['IC']:+.3f}",flush=True)
    return a,ok

print("\n===== FULL WINDOW REVALIDATE LIBRARY =====",flush=True)
lib={}
for nm,fv,fl in [('beta_VIX_60',bv,True),('kaufman_eff_20d',ka,False),('mom_