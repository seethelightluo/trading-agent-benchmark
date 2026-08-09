import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,1900)
  except:pass
  if d is not None and len(d):break
 if d is None:continue
 x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);x=x.drop_duplicates('date').sort_values('date');
 peak=x.close.shift(1).rolling(60,min_periods=40).max(); dd=x.close.shift(1)/peak-1
 x['factor']=x.close.pct_change(20)/(1+(-dd).clip(lower=0));x['fwd']=x.close.shift(-1)/x.close-1;x['symbol']=s;rows.append(x[['date','symbol','factor','fwd']])
df=pd.concat(rows);o=[]
for dt,g in df.groupby('date'):
 g=g.replace([np.inf,-np.inf],np.nan).dropna()
 if len(g)>=8:
  q=g.factor.corr(g.fwd,method='spearman')
  if np.isfinite(q):o.append((dt,q,len(g)))
o=pd.DataFrame(o,columns=['date','ic','n']);print('dates',len(o),'assets',df.symbol.nunique(),'avgN',o.n.mean(),'period',o.date.min(),o.date.max());print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 z=o[(o.date>=a)&(o.date<=b)]
 if len(z):print(a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1))
