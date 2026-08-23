import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=4000):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   d=fn(s,n)
   if d is not None and len(d): return d.assign(date=pd.to_datetime(d.date)).set_index('date')['close'].astype(float)
  except Exception: pass
 return None
px=pd.DataFrame({s:get(s) for s in U}).sort_index().ffill()
ret=px.pct_change(60).shift(1); vol=px.pct_change().rolling(20).std().shift(1); f=ret/vol
rows=[]; fr=px.pct_change(10).shift(-10)
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avgN',round(r.n.mean(),3),'coverage',round(r.n.mean()/15,4))
print('IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for n in [60,180,365,730]:
 q=r.tail(n); print('recent',n,round(q.ic.mean(),8),round(q.ic.mean()/q.ic.std(ddof=1),6),len(q))
for h in [1,3,5,10,20]:
 yy=px.pct_change(h).shift(-h); a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('h',h,'IC',round(float(np.nanmean(a)),8),'N',len(a))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'start',r.index.min(),'end',r.index.max())
f.to_csv('scripts/miner_2_20320108_volscaled_momentum_signal.csv'); r.to_csv('scripts/miner_2_20320108_volscaled_momentum_ic.csv')
