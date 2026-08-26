"""miner_3 cycle 2033-03-17: re-validate effective library + sweep new candidates.
Visible through 2033-03-16. No lookahead. Gates: abs IC>=0.0070, abs ICIR>=0.084 (10d). Warm-up only.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-03-16'
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
fwd5=rets.shift(-5).rolling(5).mean(); fwd10=rets.shift(-10).rolling(10).mean(); fwd20=rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}",flush=True)
def mac(c):
    df=pd.read_csv(ID/f'{c}.csv',parse_dates=['date']); return df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
vix=mac('VIX'); dxy=mac('DXY'); usdcny=mac('USDCNY')
dVIX=vix.pct_change(); dCNY=usdcny.pct_change(); dDXY=dxy.pct_change(); dxyr=dDXY.reindex(close.index)
def compute_ic(fv,fwd,min_dates=30,start=None):
    f=fv.reindex(fwd.index); idx=fwd.index
    if start: idx=idx[idx>=pd.Timestamp(start)]
    ics=[]; ok=0
    for d in idx:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0: ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return dict(IC=0.,ICIR=0.,n=len(ics),hit=0.,cov=0.,ok=ok)
    mu=ics.mean(); sd=ics.std(); return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),hit=float((ics>0).mean()),cov=float(f.notna().mean().mean()),ok=ok)
def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0); return float((s.diff()!=0).mean().mean())
def report(name,fv,start=None):
    fv=fv.reindex(fwd10.index); a=compute_ic(fv,fwd10,start=start); b=compute_ic(fv,fwd5,start=start); c=compute_ic(fv,fwd20,start=start)
    ok=abs(a['IC'])>=0.0070 and abs(a['ICIR'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.4f} [20]{c['IC']:.4f} [20i]{c['ICIR']:.4f}",flush=True)
    return a

print("\n===== REVALIDATE EFFECTIVE LIBRARY (full window) =====")
ac1=rets.rolling(120,min_periods=60).apply(lambda x:np.corrcoef(x[1:],x[:-1])[0,1] if len(x)>3 else np.nan,raw=False)
report("ac1_120d",ac1)
bw=rets.rolling(20).std(); report("bb_width_20d",bw)
bv=((rets.rolling(60).cov(dVIX)).div(dVIX.rolling(60).var())); report("beta_VIX_60",bv)
cb=((rets.rolling(60).cov(dCNY)).div(dCNY.rolling(60).var())); report("cny_beta_60",cb)
report("dxy_corr_change_20_60",rets.rolling(20).corr(dxyr)-rets.rolling(60).corr(dxyr))
kef=((close.diff(20).abs()).div(close.diff().abs().rolling(20).sum())); report("kaufman_eff_20d",kef)
mom10=(close.shift(5)/close.shift(15)-1.0); report("mom_10d_skip5",mom10)
mom120=(close.shift(5)/close.shift(125)-1.0); report("mom_120d_skip5",mom120)
report("skew_20d",(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3)))
volz=rets.rolling(20).std().rank(axis=1); report("vol_z_20d",volz)

print("\n=== RECENT 2Y DRIFT (2031-01-01+) ===")
RECENT='2031-01-01'
report("beta_VIX_60[r]",bv,start=RECENT)
report("kaufman_eff_20d[r]",kef,start=RECENT)
report("mom_10d_skip5[r]",mom10,start=RECENT)
report("bb_width_20d[r]",bw,start=RECENT)
report("cny_beta_60[r]",cb,start=RECENT)
report("mom_120d_skip5[r]",mom120,start=RECENT)
report("vol_z_20d[r]",volz,start=RECENT)
report("ac1_120d[r]",ac1,start=RECENT)

print("\n=== SWEEP NEW CANDIDATES ===")
report("NEW retrace_high120",close/close.rolling(120).max()-1)
report("NEW vol_ratio_10_60",rets.rolling(10).std()/rets.rolling(60).std())
dd=rets.where(rets<0,0)
report("NEW downside_vol_60",dd.rolling(60).std())
report("NEW eff_ratio_10",(close.diff(10).abs()).div(close.diff().abs().rolling(10).sum()))
report("NEW high_low_range_20",(high.rolling(20).max()-low.rolling(20).min())/close)