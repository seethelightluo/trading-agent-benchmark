import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   d=fn(s,4000)
   if d is not None and len(d): return d.assign(date=pd.to_datetime(d.date)).set_index('date')['close'].astype(float)
  except: pass
px=pd.DataFrame({s:get(s) for s in U}).sort_index().ffill(); r=px.pct_change()
mom=px.pct_change(20).shift(1); consistency=(r.rolling(60).mean()/r.abs().rolling(60).mean()).shift(1); f=mom*consistency; fr=px.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avgN',r.n.mean(),'coverage',r.n.mean()/15)
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for n in [60,180,365,730]:
 q=r.tail(n); print('recent',n,q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
for h in [1,3,5,10,20]:
 yy=px.pct_change(h).shift(-h);a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('h',h,np.nanmean(a),len(a))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'start',r.index.min(),'end',r.index.max())
f.to_csv('scripts/miner_2_20320108_consistency_momentum_signal.csv');r.to_csv('scripts/miner_2_20320108_consistency_momentum_ic.csv')
