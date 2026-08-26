"""miner_3 cycle 2033-03-31. Visible through 2033-03-30. No lookahead.
Fix beta_VIX_60/cny_beta_60 NaN from index alignment. Sweep new candidates
for library correlation + recent-window robustness. Gates: abs IC>=0.0070, abs ICIR>=0.084 (10d).
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-03-30'
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
vix=mac('VIX'); usdcny=mac('USDCNY')
dVIX=vix.pct_change(); dCNY=usdcny.pct_change()

# fast lag-1 rolling autocorrelation of returns
def roll_acorr(x, win):
    out=pd.Series(np.nan,index=x.index)
    xc=x.values
    n=len(xc)
    for i in range(win-1,n):
        seg=xc[i-win+1:i+1]
        if np.std(seg)>0 and len(seg)>3:
            out.iloc[i]=np.corrcoef(seg[1:],seg[:-1])[0,1]
    return out
# compute per asset
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
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.3f} [20]{c['IC']:.3f}",flush=True)
    return a

print("\n===== REVALIDATE EFFECTIVE LIBRARY (full window) =====",flush=True)
report("ac1_120d",ac1,flip=True)
report("bb_width_20d",rets.rolling(20).std(),flip=True)
bv=rets.rolling(60).cov(dVIX).div(dVIX.rolling(60).var()); report("beta_VIX_60",bv,flip=True)
cb=rets.rolling(60).cov(dCNY).div(dCNY.rolling(60).var()); report("cny_beta_60",cb)
report("kaufman_eff_20d",(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum()))
report("mom_10d_skip5",close.shift(5)/close.shift(15)-1.0)
report("mom_120d_skip5",close.shift(5)/close.shift(125)-1.0)
report("skew_20d",(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3)))
report("vol_z_20d",rets.rolling(20).std().rank(axis=1),flip=True)

print("\n=== RECENT 2Y DRIFT (2031-04-01+) ===",flush=True)
REC='2031-04-01'
report("beta_VIX_60[r]",bv,start=REC,flip=True)
report("kaufman_eff_20d[r]",(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum()),start=REC)
report("mom_10d_skip5[r]",close.shift(5)/close.shift(15)-1.0,start=REC)
report("bb_20d[r]",rets.rolling(20).std(),start=REC,flip=True)
report("cny_beta_60[r]",cb,start=REC)
report("mom_120d_skip5[r]",close.shift(5)/close.shift(125)-