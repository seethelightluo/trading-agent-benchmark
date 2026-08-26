import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is not None and len(d)>300:
  x=d[['date','close','volume']].dropna(subset=['date','close']).drop_duplicates('date').set_index('date')
  P[s]=x.close.astype(float)
  V[s]=x.volume.astype(float).replace(0,np.nan)
p=pd.DataFrame(P).sort_index(); v=pd.DataFrame(V).reindex(p.index)
r=p.pct_change()
# Relative 20-session reversal, amplified only by unusual but bounded volume shock; all inputs lagged one session.
rel20=r.rolling(20,min_periods=15).sum(); rel20=rel20.sub(rel20.median(axis=1),axis=0)
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
vs=(v.rolling(20,min_periods=12).mean()/v.rolling(120,min_periods=60).mean()-1).clip(-0.5,1.0)
f=((-rel20/vol.replace(0,np.nan))*(1+0.35*vs.clip(lower=0))).shift(1)
print('DATA dates',len(p),'instruments',len(P),'range',p.index.min(),p.index.max())
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-07-07')]:
   z=q.loc[a:b]
   if len(z): print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20320708_volume_shock_reversal_signal.csv',index=False)
