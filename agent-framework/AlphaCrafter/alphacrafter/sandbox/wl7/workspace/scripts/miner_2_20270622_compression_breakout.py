import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-06-21')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]; rr=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); v5=r.rolling(5,min_periods=4).std(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
 # Breakout continuation: medium return, rewarded by compressed recent volatility, lagged one session.
 f=(c.pct_change(20)/(v20*np.sqrt(20)+1e-12)*(v20/(v60+1e-12))**-0.5).shift(1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f}))
 rr.append(pd.DataFrame({'date':c.index,'asset':s,'fr':c.shift(-1).values/c.values-1}))
P=pd.concat(rows,ignore_index=True); R=pd.concat(rr,ignore_index=True); x=P.merge(R).replace([np.inf,-np.inf],np.nan).dropna(); vals=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
z=pd.Series(vals); print('assets',len(D),'dates',len(z),'avg_n',round(np.mean(ns),2),'ic',round(z.mean(),6),'icir',round(z.mean()/z.std(ddof=1)*np.sqrt(252),6),'hit',round((z>0).mean(),4),'coverage',round(len(x)/(x.date.nunique()*len(U)),4))
for h in [5,10,20]:
 y=[]
 for s,d in D.items():
  c=d.close.astype(float); f=P[P.asset==s].set_index('date').f.reindex(c.index); q=pd.DataFrame({'f':f,'fr':c.shift(-h)/c-1},index=c.index).dropna(); q['asset']=s; y.append(q.reset_index(names='date'))
 y=pd.concat(y); vv=[]
 for _,g in y.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:vv.append(g.f.corr(g.fr,method='spearman'))
 print('decay',h,'dates',len(vv),'ic',round(np.mean(vv),6))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 q=x[x.date.dt.year.between(a,b)]; v=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:v.append(g.f.corr(g.fr,method='spearman'))
 print('regime',a,b,'dates',len(v),'ic',round(np.mean(v),6),'icir',round(np.mean(v)/np.std(v,ddof=1)*np.sqrt(252),6))
r=P.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),6)); P.to_csv('scripts/miner_2_20270622_compression_breakout_signal.csv',index=False)
