import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>=100:return d
  except:pass
P=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U if g(s) is not None}).sort_index(); r=np.log(P).diff()
# medium-term momentum divided by recent realized risk, centered cross-section
f=(np.log(P/P.shift(60))/(r.rolling(20,min_periods=15).std()*np.sqrt(60))).sub((np.log(P/P.shift(60))/(r.rolling(20,min_periods=15).std()*np.sqrt(60))).mean(axis=1),axis=0)
y=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q):rows.append((dt,len(z),q))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20331208_medium_momentum_signal.csv',index=False);o.to_csv('scripts/miner_1_20331208_medium_momentum_ic.csv')
for n,s in [('full',o.ic),('365d',o.ic.tail(365)),('750d',o.ic.tail(750)),('1260d',o.ic.tail(1260))]:print(n,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
print('assets',len(P.columns),'avgN',o.n.mean(),'coverage',len(o)*o.n.mean()/(len(P.index)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]:
 z=[];yy=np.log(P.shift(-h)/P)
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(z))
