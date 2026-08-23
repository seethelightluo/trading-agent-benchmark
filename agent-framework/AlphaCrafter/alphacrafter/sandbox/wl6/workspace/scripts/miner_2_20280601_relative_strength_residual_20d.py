import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s, days=4000)
            if d is not None and len(d): return d[['date','close']].drop_duplicates('date').set_index('date')['close'].rename(s)
        except (FileNotFoundError,KeyError): pass
    raise RuntimeError('missing '+s)
px=pd.concat([fetch(s) for s in U],axis=1).sort_index().ffill(); raw=px.shift(1)/px.shift(21)-1
fac=raw.sub(raw.median(axis=1),axis=0); fwd=px.shift(-1)/px-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('range',o.index.min(),o.index.max(),'dates',len(o),'avg_n',o.n.mean(),'min_n',o.n.min())
print('daily IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1), (o.ic>0).mean()))
for h in [5,10]:
 fw=px.shift(-h)/px-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); print('%dd IC %.6f ICIR %.6f n %d'%(h,a.mean(),a.mean()/a.std(ddof=1),len(a)))
rank=fac.rank(axis=1,pct=True); print('coverage %.4f rank_turnover %.6f'%(fac.notna().mean().mean(),rank.diff().abs().mean(axis=1).mean()))
for label,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-27','2026-01-01','2027-12-31'),('2028','2028-01-01','2028-05-31')]:
 q=o.loc[a:b,'ic']; print('regime',label,'n',len(q),'IC %.6f'%q.mean())
