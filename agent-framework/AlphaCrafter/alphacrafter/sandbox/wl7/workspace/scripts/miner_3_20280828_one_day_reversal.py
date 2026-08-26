import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-08-27')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}; rows=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change();
 # bounded one-day reversal, reducing impact of extreme shocks
 f=(-r/(1+abs(r))).shift(1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1}))
q=pd.concat(rows,ignore_index=True).dropna(subset=['f','fr1'])
def stat(x,col):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1: z.append(g.f.corr(g[col],method='spearman')); ns.append(len(g))
 z=pd.Series(z).dropna(); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*len(D)))
for c in ['fr1','fr5','fr10']: print(c,stat(q,c))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]: print('regime',a,b,stat(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr1'))
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean().mean())
q.to_csv('scripts/miner_3_20280828_one_day_reversal_signal.csv',index=False)
