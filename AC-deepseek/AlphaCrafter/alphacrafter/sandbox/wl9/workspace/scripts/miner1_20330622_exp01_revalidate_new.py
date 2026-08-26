"""miner_1 cycle 2033-06-23. Visible through 2033-06-22. No lookahead.
Revalidate effective library in recent window + sweep fresh candidates.
Gates: abs IC>=0.0070, abs ICIR>=0.084 (10d), cross-section >=8 names.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-06-22'
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
def compute_ic2(f,fwd,min_dates=30,start=None):
    ii=fwd.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[]; ok=0
    for d in ii:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0: ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)
def report(name,fv,start=None,flip=False):
    f=fv.reindex(fwd10.index)
    if flip: f=-f
    a=compute_ic2(f,fwd10,start=start); b=compute_ic2(f,fwd5,start=start); c=compute_ic2(f,fwd20,start=start)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} | [5]{b['IC']:.3f}[20]{c['IC']:.3f}",flush=True)
    return a
print("\n===== REVALIDATE EFFECTIVE LIBRARY (full window) =====",flush=True)
report("beta_VIX_60",bv,flip=True); report("kaufman_eff_20d",ka); report("mom_120d_skip5",mom120)
report("bb_width_20d",bb,flip=True); report("cny_beta_60",cb); report("vol_z_20d",volz,flip=True)
report("ac1_120d",ac1,flip=True); report("mom_10d_skip5",mom10)
report("dxy_corr_change_20_60",db.diff(40),flip=True); report("skew_20d",skew)
print("\n=== RECENT DRIFT (2030-12-01+) ===",flush=True)
REC='2030-12-01'
report("beta_VIX_60[r]",bv,start=REC,flip=True); report("kaufman_eff_20d[r]",ka,start=REC); report("mom_10d_skip5[r]",mom10,start=REC)
report("bb_20d[r]",bb,start=REC,flip=True); report("cny_beta_60[r]",cb,start=REC); report("mom_120d_skip5[r]",mom120,start=REC)
report("ac1_120d[r]",ac1,start=REC,flip=True); report("vol_z_20d[r]",volz,start=REC,flip=True)
report("dxy_corr_change_20_60[r",db.diff(40),start=REC,flip=True); report("skew_20d[r]",skew,start=REC)