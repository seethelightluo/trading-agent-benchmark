import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}; px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items()}).sort_index()
# risk-adjusted intermediate momentum: 10d return / 30d realized vol, lagged
f=(px.pct_change(10)/px.pct_change().rolling(30).std()).shift(1)
fr=px.pct_change(10).shift(-10); vals=[];dates=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append(c);dates.append(dt);ns.append(len(z))
a=np.array(vals); print('dates',len(a),'avgN',np.mean(ns),'coverage',f.notna().mean().mean(),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for lo,hi in [('2020','2024'),('2025','2029'),('2030','2035')]:
 q=a[(np.array(dates)>=pd.Timestamp(lo+'-01-01'))&(np.array(dates)<=pd.Timestamp(hi+'-12-31'))];print(lo,'dates',len(q),'IC',np.mean(q) if len(q) else None,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else None)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20350216_riskadj_momentum_signal.csv',index=False)
