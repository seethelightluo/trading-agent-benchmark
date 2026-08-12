import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<150: d=get_index_daily_data(s,2800)
 return d.set_index('date').close.rename(s)
close=pd.concat([get(s) for s in U],axis=1).sort_index().ffill()
r=close.pct_change()
# Smooth breakout continuation: recent return, penalize volatility, and require price above lagged medium trend.
vol=r.rolling(40,min_periods=25).std()
trend=close.shift(1).pct_change(60)
recent=close.shift(1).pct_change(20)
# bounded trend-quality score, lagged one day by construction
f=(recent/(vol*np.sqrt(40)+1e-8))*np.tanh(trend/0.20)
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(close)-20):
 x=f.iloc[i]
 if x.notna().sum()<8: continue
 for h in [1,5,10,20]:
  y=close.iloc[i+h]/close.iloc[i]-1
  z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((close.index[i],h,len(z),z.x.corr(z.y)))
rp=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('range',close.index.min(),close.index.max(),'assets',len(close.columns),'dates',rp.date.nunique())
for h in [1,5,10,20]:
 q=rp[rp.h==h].groupby('date').ic.first().dropna(); print(h,'dates',len(q),'avgN',rp[rp.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for yr in [2024,2025,2026,2027,2028,2029,2030]:
 q=rp[(rp.h==10)&(rp.date.dt.year==yr)].groupby('date').ic.first().dropna()
 if len(q)>1: print('yr',yr,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
sig=f.rank(axis=1,pct=True)
print('coverage',f.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
f.to_csv('scripts/miner_1_20301031_breakout_quality_signal.csv')
