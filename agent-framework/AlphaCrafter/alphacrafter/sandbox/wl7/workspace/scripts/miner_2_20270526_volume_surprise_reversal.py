import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-25')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
# Volume-surprise reversal: lagged 3d reversal is emphasized when the move occurred on unusual volume.
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=pd.to_numeric(d.volume,errors='coerce')
 vs=np.log((vol+1)/(vol.rolling(20,min_periods=10).median()+1))
 f=-(c/c.shift(3)-1)*vs.clip(-2,2)/(r.rolling(20,min_periods=15).std()+1e-12)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.shift(1),'fr':c.shift(-1)/c-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[];ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:z.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 z=pd.Series(z).dropna(); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*15),'daily',stats(q))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(p.diff().abs().mean().mean()))
for h in [5,10]:
 rr=[]
 for s,d in D.items():
  c=d.close.astype(float); rr.append(pd.DataFrame({'date':c.index,'asset':s,'f':(-(c/c.shift(3)-1)*np.log((pd.to_numeric(d.volume,errors='coerce')+1)/(pd.to_numeric(d.volume,errors='coerce').rolling(20,min_periods=10).median()+1))).clip(-2,2).shift(1),'fr':c.shift(-h)/c-1}))
 x=pd.concat(rr,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(); print('decay',h,stats(x))
q.to_csv('scripts/miner_2_20270526_volume_surprise_reversal_signal.csv',index=False)
