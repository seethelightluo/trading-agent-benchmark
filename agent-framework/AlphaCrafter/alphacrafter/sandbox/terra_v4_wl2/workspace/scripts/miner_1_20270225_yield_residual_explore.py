import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
y=r[['US10Y','CN10Y']].rolling(5).sum(); yz=y.mean(axis=1); yshock=((yz-yz.rolling(120,min_periods=60).mean())/yz.rolling(120,min_periods=60).std()).shift(1)
mom=r.rolling(20).sum(); rel=mom.sub(mom.median(axis=1),axis=0)
cond=pd.DataFrame(np.tile(np.where(yshock.reindex(rel.index).values[:,None]>0,-1,1),(1,len(rel.columns))),index=rel.index,columns=rel.columns)
spread=(y['US10Y']-y['CN10Y']).shift(1); cond2=pd.DataFrame(np.tile(np.where(spread.reindex(rel.index).values[:,None]>0,-1,1),(1,len(rel.columns))),index=rel.index,columns=rel.columns)
resid=rel-yshock.reindex(rel.index).values[:,None]*rel*0.05
for name,fac in [('yield_cond_relmom',rel*cond),('yield_spread_relmom',rel*cond2),('yield_neutral_resid',resid)]:
 print('\n',name)
 fr=px.shift(-5)/px-1; arr=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   arr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 a=np.array(arr); print('dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',np.mean(ns)/15)
 for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2026-07+','2026-07-16','2027-02-25')]:
  q=[v for d,v in zip(dates,a) if str(d)>=lo and str(d)<=hi]; print(lab,len(q),np.mean(q) if q else np.nan,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
 print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
 out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_'+name+'.csv',index=False)
