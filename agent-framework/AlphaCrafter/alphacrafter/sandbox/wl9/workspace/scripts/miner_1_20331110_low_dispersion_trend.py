import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>300: cl[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(cl).sort_index(); r=p.pct_change()
disp=r.std(axis=1).rolling(20,min_periods=15).mean()
rank=disp.rolling(252,min_periods=126).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],raw=False)
mom=p.pct_change(60)/(r.rolling(60,min_periods=45).std()*np.sqrt(252)+.05)
f=mom.where(rank<.40,0).clip(-5,5).shift(1)
print('DATA',len(p),len(cl),p.index.min(),p.index.max())
for h in [10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna(); print('H',h,'dates',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2029-12-31'),('2030','2030-12-31'),('2031','2032-12-31'),('2033','2033-11-10')]:
   z=q.loc[a:b]
   if len(z): print('REGIME',a,len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%((f!=0).mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20331110_low_dispersion_trend_signal.csv',index=False)
