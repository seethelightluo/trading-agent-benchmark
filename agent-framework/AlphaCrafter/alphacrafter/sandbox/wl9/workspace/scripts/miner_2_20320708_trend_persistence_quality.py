import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is not None and len(d)>300: C[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change()
# Trend persistence quality: medium horizon return, rewarded when short and medium trends agree;
# risk scale by realized volatility, and neutralize cross-section. Lag to avoid lookahead.
r20=p.pct_change(20); r60=p.pct_change(60); r5=p.pct_change(5)
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
agree=np.sign(r20)*np.sign(r60)
# continuous agreement: medium trend plus short confirmation, volatility normalized
raw=(0.55*r20+0.25*r60+0.20*r5)*agree/vol.replace(0,np.nan)
f=raw.sub(raw.median(axis=1),axis=0).shift(1)
print('DATA dates',len(p),'instruments',len(C),'range',p.index.min(),p.index.max())
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
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
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20320708_trend_persistence_quality_signal.csv',index=False)
