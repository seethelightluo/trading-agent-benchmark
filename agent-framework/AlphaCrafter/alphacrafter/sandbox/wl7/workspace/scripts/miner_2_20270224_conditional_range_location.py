import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-24')
def F(s):
 for fn in [get_index_daily_data,get_stock_daily_data]:
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:F(s) for s in U}; D={s:x for s,x in D.items() if x is not None}; P=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index()
# Conditional low-range-location reversal: only activate when lagged cross-asset dispersion is elevated.
R=P.pct_change(); disp=R.rolling(20,min_periods=15).std().mean(axis=1); threshold=disp.rolling(252,min_periods=100).median()
rows=[]
for s in D:
 hi=P[s].rolling(20,min_periods=15).max(); lo=P[s].rolling(20,min_periods=15).min()
 loc=(-(P[s]-lo)/(hi-lo+1e-12)).shift(1)
 f=loc.where(disp.shift(1)>threshold.shift(1))
 rows.append(pd.DataFrame({'date':P.index,'asset':s,'f':f,'fr1':P[s].shift(-1)/P[s]-1,'fr5':P[s].shift(-5)/P[s]-1,'fr10':P[s].shift(-10)/P[s]-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan)
def stats(x,col):
 x=x.dropna(subset=['f',col]); z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:z.append(g.f.corr(g[col],method='spearman'));ns.append(len(g))
 z=pd.Series(z); return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('assets',len(D),'dates',q.date.nunique(),'rows',len(q),'coverage',round(q.f.notna().mean(),4))
for c in ['fr1','fr5','fr10']: print(c,stats(q,c))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr1'))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',round(float(r.diff().abs().mean().mean()),5))
q.to_csv('scripts/miner_2_20270224_conditional_range_location_signal.csv',index=False)
