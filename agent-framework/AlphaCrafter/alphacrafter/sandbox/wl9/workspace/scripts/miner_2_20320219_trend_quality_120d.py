import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4400)
 if d is not None and len(d)>300: C[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change(); lr=np.log(p); rv=r.rolling(40,min_periods=20).std()*np.sqrt(40)
x=np.arange(120); fs={}
for s in p:
 y=lr[s]
 def sl(z):
  xx=x[-len(z):]; return ((xx-xx.mean())*(z-z.mean())).sum()/(((xx-xx.mean())**2).sum()+1e-12)
 slope=y.rolling(120,min_periods=100).apply(sl,raw=True)
 fitvol=(y-y.rolling(120,min_periods=100).mean()).rolling(120,min_periods=100).std()
 fs[s]=(slope*120/(fitvol+1e-8)/(rv[s]+1e-8)).shift(1)
f=pd.DataFrame(fs); print('loaded',len(C),'assets dates',len(p))
for h in [10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna(); print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20320219_trend_quality_120d_signal.csv',index=False)
