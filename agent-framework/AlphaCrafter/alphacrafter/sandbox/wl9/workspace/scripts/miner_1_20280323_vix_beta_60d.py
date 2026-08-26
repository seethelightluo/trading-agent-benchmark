import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for root in ['../persistent/stock_data/','../persistent/index_data/']:
  f=root+s+'.csv'
  if os.path.exists(f): return pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close.astype(float)
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
V=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close
# negative rolling VIX beta, only past observations
F={}
for s,p in D.items():
 z=pd.concat([p.pct_change().rename('r'),V.pct_change().rename('v')],axis=1).dropna()
 F[s]=(-z.r.rolling(60,min_periods=45).cov(z.v)/z.v.rolling(60,min_periods=45).var()).reindex(p.index)
F=pd.DataFrame(F); rows=[]
for dt in F.index:
 xs=[]; ns=[]
 for s,p in D.items():
  if dt not in p.index or pd.isna(F.loc[dt,s]): continue
  i=p.index.get_loc(dt)
  for h in [1,5,10]:
   pass
 # use each asset's next valid observation, avoiding calendar mismatch
 for h in [1,5,10]:
  x=[];y=[]
  for s,p in D.items():
   if dt in p.index and pd.notna(F.loc[dt,s]):
    i=p.index.get_loc(dt)
    if i+h<len(p) and pd.notna(p.iloc[i+h]): x.append(F.loc[dt,s]); y.append(p.iloc[i+h]/p.iloc[i]-1)
  if len(x)>=8 and len(set(x))>1: rows.append((dt,h,spearmanr(x,y).statistic,len(x)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=r[r.h==h]
 print('h',h,'dates',len(q),'meanN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for label,sub in [('online',q[q.date>=pd.Timestamp('2026-07-16')]),('recent',q[q.date>=pd.Timestamp('2027-03-23')])]:
  print(label,'dates',len(sub),'IC',round(sub.ic.mean(),6) if len(sub) else None,'ICIR',round(sub.ic.mean()/sub.ic.std(ddof=1),6) if len(sub)>1 else None)
print('assets',len(D),'coverage',round(F.notna().sum(axis=1).mean()/len(D),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,5))
