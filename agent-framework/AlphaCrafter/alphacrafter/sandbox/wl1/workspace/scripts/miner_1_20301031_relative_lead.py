import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<150:d=get_index_daily_data(s,2800)
 return d.set_index('date').close.rename(s)
c=pd.concat([get(s) for s in U],axis=1).sort_index().ffill(); r=c.pct_change(); lag=c.shift(1)
# Relative leadership: asset 20d return versus contemporaneous cross-sectional median,
# stabilized by its own 60d volatility and smoothed with 10d EWMA.
raw=lag.pct_change(20).sub(lag.pct_change(20).median(axis=1),axis=0)
vol=r.rolling(60,min_periods=30).std(); f=(raw/(vol*np.sqrt(60)+1e-8)).ewm(span=10,min_periods=5).mean()
rows=[]
for i in range(len(c)-20):
 x=f.iloc[i]
 for h in [1,5,10,20]:
  y=c.iloc[i+h]/c.iloc[i]-1; z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((c.index[i],h,len(z),z.x.corr(z.y)))
rp=pd.DataFrame(rows,columns=['date','h','n','ic']); print('range',c.index.min(),c.index.max(),'dates',rp.date.nunique(),'assets',len(c.columns))
for h in [1,5,10,20]:
 q=rp[rp.h==h].groupby('date').ic.first().dropna(); print(h,'dates',len(q),'avgN',rp[rp.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for yr in [2024,2025,2026,2027,2028,2029,2030]:
 q=rp[(rp.h==10)&(rp.date.dt.year==yr)].groupby('date').ic.first().dropna()
 if len(q)>1: print('yr',yr,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_1_20301031_relative_lead_signal.csv')
