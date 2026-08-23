import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(h):
 rows=[]
 for s in U:
  f='../persistent/stock_data/'+s+'.csv'
  if not os.path.exists(f): continue
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=pd.Timestamp('2028-05-03')].copy(); r=d.close.pct_change()
  sig=(r.rolling(5).sum()-r.rolling(20).sum()/4).shift(1)/r.rolling(20).std().shift(1); fw=d.close.shift(-h)/d.close-1
  rows += list(zip(d.date, [s]*len(d), sig, fw))
 return pd.DataFrame(rows,columns=['date','symbol','signal','fwd']).replace([np.inf,-np.inf],np.nan).dropna()
def calc(z):
 vals=[]; meta=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:
   vals.append(spearmanr(g.signal,g.fwd).statistic); meta.append((dt,len(g)))
 return np.array(vals),meta
z=get(1); a,m=calc(z); print('dates',len(a),'rows',len(z),'avg_n',np.mean([x[1] for x in m]),'coverage',len(z)/(len(U)*len(m)),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2026-12-31'),('2027-01-01','2027-12-31'),('2028-01-01','2028-05-03')]:
 q=np.array([v for (dt,n),v in zip(m,a) if pd.Timestamp(lo)<=dt<=pd.Timestamp(hi)]); print(lo,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 q,mh=calc(get(h)); print('h',h,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1))
z[['date','symbol','signal']].to_csv('scripts/miner_3_20280504_accel_signal.csv',index=False)
