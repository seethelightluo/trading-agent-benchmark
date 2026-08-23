import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-09')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize()
    return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
P=pd.concat({s:d.close for s,d in D.items()},axis=1).sort_index(); R=P.pct_change()
disp=R.rolling(5,min_periods=3).std().mean(axis=1)
threshold=disp.rolling(60,min_periods=30).median().shift(1)
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
 f=(-(c.pct_change(5))/(vol+1e-12)).where(disp.reindex(c.index).shift(1)>threshold.reindex(c.index))
 for h in [1,5,10,20]: pass
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1,'fr20':c.shift(-20)/c-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan)
def stats(x,col):
 z=[]; ns=[]
 for _,g in x[['date','asset','f',col]].dropna().groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1: z.append(g.f.corr(g[col],method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('asof',CUT.date(),'assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',q.groupby('date').f.count().mean(),'coverage',q.f.notna().mean())
for h in [1,5,10,20]: print('horizon',h,stats(q,f'fr{h}'))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr1'))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_2_20270309_dispersion_reversal_signal.csv',index=False)
