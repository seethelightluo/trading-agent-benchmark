import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): cl[s]=d.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(cl).sort_index().ffill()
# Short-horizon mean reversion: negative recent 5-day return, damped by 20-day volatility.
r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
raw=-r.rolling(5).sum()/(v*np.sqrt(5)+1e-12)
sig=raw.rank(axis=1,pct=True).shift(1); rows=[]
for i in range(len(p)-21):
 for h in [1,5,10,20]:
  z=pd.concat([sig.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((p.index[i],h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']); print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('year1',o[o.h==1].set_index('date').ic.groupby(lambda x:x.year).mean().round(5).to_dict())
sig.to_csv('scripts/miner_3_20310220_short_reversal_signal.csv')
