import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 x['fac']=-(x.open/x.close.shift(1)-1); x['asset']=s
 for h in [1,3,5,10]: x['f'+str(h)]=x.close.shift(-h)/x.close-1
 rows.append(x[['date','asset','fac']+['f'+str(h) for h in [1,3,5,10]]])
a=pd.concat(rows)
for h in [1,3,5,10]:
 vals=[]; ns=[]
 for dt,g in a[['date','asset','fac','f'+str(h)]].dropna().groupby('date'):
  if len(g)>=8 and g.fac.nunique()>1 and g['f'+str(h)].nunique()>1:
   vals.append(spearmanr(g.fac,g['f'+str(h)]).statistic); ns.append(len(g))
 v=np.array(vals)
 print('h',h,'dates',len(v),'avg_n',np.mean(ns),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
for lo,hi in [('2020','2025'),('2026','2029'),('2030','2033')]:
 g=a[(a.date>=lo)&(a.date<hi)]
 vals=[]
 for dt,z in g[['date','fac','f1']].dropna().groupby('date'):
  if len(z)>=8 and z.fac.nunique()>1 and z.f1.nunique()>1: vals.append(spearmanr(z.fac,z.f1).statistic)
 v=np.array(vals); print(lo,hi,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
print('coverage',a.fac.notna().mean(),'assets',len(U),'total dates',a.date.nunique())
out=a[['date','asset','fac']].dropna().rename(columns={'asset':'symbol','fac':'signal'});out.to_csv('scripts/miner_2_20330318_gap_reversal_signal.csv',index=False)
