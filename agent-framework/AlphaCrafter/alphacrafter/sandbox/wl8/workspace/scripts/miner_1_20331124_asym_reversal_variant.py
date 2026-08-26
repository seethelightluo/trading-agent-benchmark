import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 try:d=get_index_daily_data(s,5000)
 except:d=None
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,5000)
  except:d=None
 return d
P={s:g(s) for s in U}; P={s:d.set_index('date').close for s,d in P.items() if d is not None and len(d)};P=pd.DataFrame(P).sort_index();r=np.log(P).diff(); dn=(-r.clip(upper=0)).rolling(30,min_periods=15).std(); f=(-np.log(P/P.shift(5))/(dn.rolling(5,min_periods=3).mean()*np.sqrt(5))).rolling(3,min_periods=2).mean(); fr=np.log(P.shift(-10)/P)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q):rows.append((dt,len(z),q))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20331124_asym_reversal_variant_signal.csv',index=False);o.to_csv('scripts/miner_1_20331124_asym_reversal_variant_ic.csv')
for n,s in [('full',o.ic),('365d',o.ic.tail(365)),('750d',o.ic.tail(750)),('1260d',o.ic.tail(1260))]:print(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean())
print('assets',len(P.columns),'avgN',o.n.mean(),'coverage',len(o)*o.n.mean()/(len(P)*15),'turn',f.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]:
 y=np.log(P.shift(-h)/P);q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(q))
