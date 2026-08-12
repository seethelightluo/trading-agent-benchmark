import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); frames[s]=d.drop_duplicates('date').set_index('date').close
px=pd.concat(frames,axis=1).sort_index().ffill()
r=px.pct_change()
# benchmark and residual returns
bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
# smooth activation: dispersion relative to trailing median, clipped to avoid extreme concentration
disp=r.std(axis=1)
med=disp.rolling(120,min_periods=60).median()
activation=((disp/med)-0.85).clip(lower=0,upper=1.5)
# residual reversal, volatility normalized; all inputs through t, forecast t+1..t+10
f=(-resid.rolling(5,min_periods=5).sum()).div(resid.rolling(20,min_periods=15).std()).mul(activation,axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=px.shift(-10).div(px)-1
ics=[]; dates=[]; nobs=[]; turnovers=[]; prev=None
for dt in f.index:
    x=f.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ics.append(x[ok].corr(y[ok],method='spearman')); dates.append(dt); nobs.append(ok.sum())
        rank=x[ok].rank(pct=True)
        if prev is not None: turnovers.append((rank-prev.reindex(rank.index)).abs().mean())
        prev=rank
z=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
print('rows',len(px),'dates',len(z),'avg_n',np.mean(nobs),'coverage',np.mean(nobs)/len(U),'turnover',np.mean(turnovers))
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
for lo,hi in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-12-31')]:
 q=z.loc[lo:hi]; print(lo, 'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [1,5,10,20]:
 yy=px.shift(-h).div(px)-1; zz=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8: zz.append(f.loc[dt,ok].corr(yy.loc[dt,ok],method='spearman'))
 print('decay',h,np.nanmean(zz),len(zz))
print('last',px.index.max())
# save signal artifact for deterministic provenance
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20320819_smooth_dispersion_signal.csv')
