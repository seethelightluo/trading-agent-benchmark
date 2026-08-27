import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in [get_index_daily_data,get_stock_daily_data]:
  try:
   d=f(s,days=6000)
   if d is not None and len(d): return d.set_index('date').close.astype(float)
  except: pass
P=pd.DataFrame({s:get(s) for s in U}).sort_index().ffill(); r=P.pct_change()
ret=P/P.shift(10)-1; down=r.where(r<0).rolling(30,min_periods=15).std(); med=ret.median(axis=1)
F=(-(ret.sub(med,axis=0))/(down+1e-8)).shift(1)
F.to_csv('scripts/miner_1_20350621_residual_downside_reversal_signal.csv',index_label='date')
print('rows',len(P),'N',len(P.columns),'range',P.index.min(),P.index.max())
for h in [5,10,20,40,60]:
 fw=P.shift(-h)/P-1;q=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 q=pd.Series(q).dropna();print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'dates',len(q),'avgN',np.mean(ns))
print('coverage',F.notna().mean().mean(),'turnover10',F.rank(pct=True).diff(10).abs().mean().mean())
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=[]
 for dt in F.loc[a:b].index:
  z=pd.concat([F.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print('regime',a,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'dates',len(q))
