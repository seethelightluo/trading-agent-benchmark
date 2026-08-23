import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-07-06')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize()
    return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); v=pd.to_numeric(d.volume,errors='coerce')
 # volume confirmation is bounded and only amplifies positive/negative trend modestly
 vr=(v/(v.rolling(60,min_periods=20).median()+1e-12)).clip(0.25,4.0)
 f=(r.rolling(20,min_periods=15).sum()*(0.5+0.5*np.log1p(vr))).shift(1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr10'])
def stats(x,col):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:
   z.append(g.f.corr(g[col],method='spearman')); ns.append(len(g))
 z=pd.Series(z).dropna(); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)) if len(z)>1 else np.nan,float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for col in ['fr1','fr5','fr10']: print(col,stats(q,col))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr10'))
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(p.diff().abs().mean().mean()))
q.to_csv('scripts/miner_1_20270707_volume_confirmed_momentum_signal.csv',index=False)
