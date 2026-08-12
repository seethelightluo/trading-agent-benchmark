import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','open','high','low','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
z=pd.concat(rows)
# Close location value: close relative to daily range, averaged 10 sessions; lagged.
z['clv']=((2*z.close-z.high-z.low)/(z.high-z.low).replace(0,np.nan)).clip(-1,1)
cl=z.pivot(index='date',columns='symbol',values='close').sort_index().ffill()
f=z.pivot(index='date',columns='symbol',values='clv').sort_index().rolling(10,min_periods=7).mean().shift(1)
res={h:[] for h in [1,3,5,10]}
for dt in cl.index:
 for h in res:
  q=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): res[h].append(v)
for h,a in res.items():
 q=pd.Series(a); print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]:
 a=[]
 for dt in cl.index[mask]:
  q=pd.concat([f.loc[dt],cl.shift(-5).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): a.append(v)
 print('regime',name,'n',len(a),'IC',np.mean(a) if a else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20280824_clv10_signal.csv',index=False)
