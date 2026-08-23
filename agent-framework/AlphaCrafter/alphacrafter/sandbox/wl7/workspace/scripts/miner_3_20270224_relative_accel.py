import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-23')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Relative momentum: lagged 20d asset return minus same-date cross-sectional median,
# with a 5d recent acceleration term. All inputs end no later than prior completed day.
raw=[]
for s,d in D.items():
 c=d.close.astype(float); raw.append(pd.DataFrame({'date':c.index,'asset':s,'r20':c.pct_change(20),'r5':c.pct_change(5)}))
a=pd.concat(raw).reset_index(drop=True)
med=a.pivot(index='date',columns='asset',values='r20').median(axis=1)
a['f']=(a.r20-a.date.map(med))+0.35*(a.r5-a.date.map(a.pivot(index='date',columns='asset',values='r5').median(axis=1)))
a['f']=a.groupby('asset').f.shift(1)
fr=[]
for s,d in D.items(): fr.append(pd.DataFrame({'date':d.index,'asset':s,'fr':(d.close.shift(-1)/d.close-1).values}).reset_index(drop=True))
q=a.merge(pd.concat(fr),on=['date','asset']).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for h in [1,5,10,20]:
 rr=[]
 for s,d in D.items(): rr.append(pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index).values,'fr':(d.close.shift(-h)/d.close-1).values}).reset_index(drop=True))
 print('horizon',h,stats(pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna()))
for a1,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a1,b,stats(q[(q.date.dt.year>=a1)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_3_20270224_relative_accel_signal.csv',index=False)
