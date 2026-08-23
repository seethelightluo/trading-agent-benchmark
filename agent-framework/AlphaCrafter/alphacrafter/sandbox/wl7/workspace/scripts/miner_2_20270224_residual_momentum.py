import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-24')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
P=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); R=P.pct_change(); bench=R.mean(axis=1); rows=[]
for s in D:
 r=R[s]; cov=r.rolling(60,min_periods=40).cov(bench); bv=bench.rolling(60,min_periods=40).var(); beta=cov/(bv+1e-12)
 resid=r-beta*bench
 # residual trend, lagged one day
 f=resid.rolling(20,min_periods=15).sum()/(resid.rolling(20,min_periods=15).std()+1e-12)
 fr=P[s].shift(-1)/P[s]-1
 rows.append(pd.DataFrame({'date':P.index,'asset':s,'f':f.shift(1),'fr':fr}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 vals=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals)
 return len(z),round(float(np.mean(ns)),2),round(float(z.mean()),5),round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),4),round(float((z>0).mean()),4)
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',round(q.groupby('date').size().mean(),2),'coverage',round(len(q)/(q.date.nunique()*len(D)),4))
for h in [1,5,10,20]:
 if h==1:x=q
 else:
  xx=[]
  for s in D: xx.append(pd.DataFrame({'date':P.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(P.index).values,'fr':P[s].shift(-h)/P[s]-1}))
  x=pd.concat(xx,ignore_index=True).dropna()
 print('horizon',h,stats(x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(float(r.diff().abs().mean().mean()),5))
q.to_csv('scripts/miner_2_20270224_residual_momentum_signal.csv',index=False)
