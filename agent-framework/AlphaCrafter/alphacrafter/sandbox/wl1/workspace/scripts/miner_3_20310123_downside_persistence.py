import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): close[s]=d.drop_duplicates('date').set_index('date')['close']
p=pd.DataFrame(close).sort_index().ffill(); r=p.pct_change()
down=r.clip(upper=0).pow(2).rolling(60,min_periods=40).mean().pow(.5)
persist=(r.gt(0).rolling(40,min_periods=30).mean()-0.5)*2
raw=(p.pct_change(40)/(down*np.sqrt(40)+1e-12))*persist
signal=raw.rank(axis=1,pct=True).shift(1)
rows=[]
for i in range(len(p)-21):
 for h in [1,5,10,20]:
  z=pd.concat([signal.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((p.index[i],h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']); print('date_range',p.index.min(),p.index.max(),'dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',signal.notna().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('turnover',signal.diff().abs().mean().mean())
q20=o[o.h==20]
for y,g in q20.groupby(q20.date.dt.year):
 q=g.groupby('date').ic.first(); print('year20',y,'n',len(q),'IC %.5f ICIR %.4f'%(q.mean(),q.mean()/q.std(ddof=1)))
signal.to_csv('scripts/miner_3_20310123_downside_persistence_signal.csv')
