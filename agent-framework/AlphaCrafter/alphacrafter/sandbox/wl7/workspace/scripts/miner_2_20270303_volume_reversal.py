import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-03')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); v=pd.to_numeric(d.volume,errors='coerce')
 # Volume-confirmed short-term reversal: lagged 3d price shock, amplified by unusual volume,
 # normalized by recent volatility. All inputs are shifted before forecasting.
 vz=((v/(v.rolling(20,min_periods=10).median()+1e-12)).clip(0.25,4.0)-1.0)
 vol=r.rolling(20,min_periods=15).std()
 f=((-r.rolling(3,min_periods=3).sum())*(1+vz).clip(0.25,3.0)/(vol+1e-12)).shift(1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def calc(x,h):
 z=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),np.mean(z>0)
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for h in [1,5,10,20]:
 rr=[]
 for s,d in D.items(): rr.append(pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index).values,'fr':d.close.shift(-h)/d.close-1}).reset_index(drop=True))
 x=pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna(); print('horizon',h,calc(x,h))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 x=q[(q.date.dt.year>=a)&(q.date.dt.year<=b)].copy(); rr=[]
 for s,d in D.items(): rr.append(pd.DataFrame({'date':d.index,'asset':s,'f':x[x.asset==s].set_index('date').f.reindex(d.index).values,'fr':d.close.shift(-1)/d.close-1}).reset_index(drop=True))
 print('regime',a,b,calc(pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna(),1))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_2_20270303_volume_reversal_signal.csv',index=False)
