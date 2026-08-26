"""miner_2 cycle 2033-06-09. Visible through 2033-06-08. No lookahead.
Revalidate key effective library for drift and sweep new candidate factors.
Gates: abs IC >= 0.0070 and abs ICIR >= 0.084 (10d horizon).
Small 15-asset cross-asset universe; report dates/instruments used explicitly.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-06-08'
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
openpx=close.copy()
def mac(c):
    df=pd.read_csv(ID/f'{c}.csv',parse_dates=['date']); return df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float).reindex(idx)
vix=mac('VIX'); dysx=mac('DXY'); jpy=mac('USDJPY'); eur=mac('EURUSD'); cny=mac('USDCNY')
dVIX=vix.pct_change(); dJPY=jpy.pct_change(); dEUR=eur.pct_change(); dCNY=cny.pct_change(); dDXY=dysx.pct_change()
dxyr=dDXY.reindex(idx)

def compute_ic(fv,fw,min_dates=30,start=None,flip=False):
    f=fv.reindex(fw.index)
    if flip: f=-f
    ii=fw.index
    if start: ii=ii[ii>=pd.Timestamp(start)]
    ics=[]; ok=0
    for d in ii:
        x=f.loc[d]; y=fw.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),
                hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)
def turnover(fd):
    s=np.sign(fd.rank(axis=1).sub(fd.shape[1]/2)).fillna(0); return float((s.diff()!=0).mean().mean())
def report(name,fd,start=None,flip=False):
    fw=fwd10
    f=fd.reindex(fw.index)
    a=compute_ic(f,fw,start=start,flip=flip)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(f):.3f}",flush=True)
    return a

print("===== REVALIDATE KEY EFFECTIVE LIBRARY (full window, 10d) =====",flush=True)
ac1=rets.rolling(120,min_periods=60).apply(lambda x: np.corrcoef(x[1:],x[:-1])[0,1] if len(x)>3 and np.std(x[:-1])>0 else np.nan,raw=True)
report("ac1_120d",ac1,flip=True)
bw=rets.rolling(20).std(); report("bb_width_20d",bw)
bv=((rets.rolling(60).cov(dVIX)).div(dVIX.rolling(60).var())); report("beta_VIX_60",bv,flip=True)
cb=((rets.rolling(60).cov(dCNY)).div(dCNY.rolling(60).var())); report("cny_beta_60",cb)
report("dxy_corr_change_20_60",rets.rolling(20).corr(dxyr)-rets.rolling(60).corr(dxyr))
kef=((close.diff(20).abs()).div(close.diff().abs().rolling(20).sum())); report("kaufman_eff_20d",kef)
mom10=(close.shift(5)/close.shift(15)-1.0); report("mom_10d_skip5",mom10)
mom120=(close.shift(5)/close.shift(125)-1.0); report("mom_120d_skip5",mom120)
report("skew_20d",(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3)))
volz=rets.rolling(20).std().rank(axis=1); report("vol_z_20d",volz,flip=True)
report("days_since_high_60",close.rolling(60).apply(lambda x: len(x)-np.argmax(x) if len(x)>=30 else np.nan,raw=True))

print("===== RECENT 2Y DRIFT (2031-01-01+) =====",flush=True)
RECENT='2031-01-01'
report("beta_VIX_60[r]",bv,start=RECENT,flip=True)
report("kaufman_eff_20d[r]",kef,start=RECENT)
report("mom_120d_skip5[r]",mom120,start=RECENT)
report("bb_width_20d[r]",bw,start=RECENT)
report("cny_beta_60[r]",cb,start=RECENT)
report("vol_z_20d[r]",volz,start=RECENT,flip=True)
report("mom_10d_skip5[r]",mom10,start=RECENT)
report("ac1_120d[r]",ac1,start=RECENT,flip=True)