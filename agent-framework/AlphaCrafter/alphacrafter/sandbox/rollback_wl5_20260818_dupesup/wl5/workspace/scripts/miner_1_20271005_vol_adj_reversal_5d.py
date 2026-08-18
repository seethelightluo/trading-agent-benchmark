import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-10-05'); START=pd.Timestamp('2020-01-01'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 try: D[s]=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
 except Exception as e: print('missing',s,e)
def fac(x):
 r=x.close.pct_change(); vol=r.rolling(20,min_periods=15).std()
 return -x.close.pct_change(5)/(vol*np.sqrt(5)+1e-12)
rec=[]
for s,x in D.items():
 f=fac(x)
 for i,dt in enumerate(x.index[:-1]):
  if START<=dt<=END and pd.notna(f.iloc[i]) and pd.notna(x.close.iloc[i+1]) and x.close.iloc[i]!=0:
   rec.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
a=pd.DataFrame(rec,columns=['date','symbol','factor','fwd'])
ics=[]; ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:
  ics.append(spearmanr(g.factor,g.fwd).statistic); ns.append(len(g))
z=np.asarray(ics); print('candidate vol_adj_reversal_5d dates',len(z),'instruments',a.symbol.nunique(),'avgN',np.mean(ns),'coverage',a.symbol.nunique()/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-10-05')]:
 q=a[(a.date>=lo)&(a.date<=hi)]; zz=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: zz.append(spearmanr(g.factor,g.fwd).statistic)
 zz=np.asarray(zz); print(lo,hi,'dates',len(zz),'IC',zz.mean() if len(zz) else np.nan,'ICIR',zz.mean()/zz.std(ddof=1) if len(zz)>1 else np.nan)
# 5d and 10d decay, aligned from same signal date
for h in [5,10]:
 rec2=[]
 for s,x in D.items():
  f=fac(x)
  for i in range(len(x)-h):
   dt=x.index[i]
   if START<=dt<=END and pd.notna(f.iloc[i]): rec2.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+h]/x.close.iloc[i]-1)))
 q=pd.DataFrame(rec2,columns=['date','symbol','factor','fwd']); zz=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: zz.append(spearmanr(g.factor,g.fwd).statistic)
 print('decay',h,'d IC',np.mean(zz))
