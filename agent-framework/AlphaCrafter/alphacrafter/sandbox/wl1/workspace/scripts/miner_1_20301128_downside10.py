import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; xs=[]
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 xs.append(d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.rename(s))
p=pd.concat(xs,axis=1).sort_index().ffill(); r=p.pct_change()
# Medium trend rewarded only when downside risk is contained: 10d return divided by 30d downside deviation.
down=r.where(r<0,0).rolling(30,min_periods=20).std()
sig=(r.rolling(10,min_periods=10).sum()/(down+1e-8)).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-21]):
 for h in [1,5,10,20]:
  z=pd.concat([sig.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:rows.append((dt,h,z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','ic'])
print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',sig.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for y,g in o[o.h==1].groupby(o[o.h==1].date.dt.year):
 q=g.groupby('date').ic.first(); print('yr',y,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(pct=True).diff().abs().mean().mean())
sig.to_csv('scripts/miner_1_20301128_downside10_signal.csv')
