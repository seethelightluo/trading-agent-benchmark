import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-07-15')
def read(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();return x.loc[x.index<=cut,'close']
P={s:read(s) for s in U}; dates=sorted(set().union(*[set(x.index) for x in P.values()]));p=pd.concat(P,axis=1).reindex(dates)
# Calculate each asset on its own completed observations, avoiding calendar-gap dilution.
f=pd.concat({s:-P[s].pct_change().rolling(20,min_periods=15).std() for s in U},axis=1).reindex(dates)
def calc(h):
 z=[];ns=[]
 for d in dates:
  vals=[];ys=[]
  for s in U:
   x=P[s]
   if d in x.index:
    pos=x.index.get_loc(d)
    if pos>=20 and pos+h<len(x): vals.append(f.loc[d,s]);ys.append(x.iloc[pos+h]/x.iloc[pos]-1)
  q=pd.DataFrame({'f':vals,'y':ys}).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
 return np.array(z),ns
a,ns=calc(1);print('idea low-volatility 20d; horizon 1; dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=[]
 for d in dates:
  if lo<=d.year<=hi:
   vals=[];ys=[]
   for s in U:
    x=P[s]
    if d in x.index:
     pos=x.index.get_loc(d)
     if pos>=20 and pos+1<len(x):vals.append(f.loc[d,s]);ys.append(x.iloc[pos+1]/x.iloc[pos]-1)
   q=pd.DataFrame({'f':vals,'y':ys}).dropna()
   if len(q)>=8:z.append(spearmanr(q.f,q.y).statistic)
 z=np.array(z);print('regime',lo,hi,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for h in [5,10]:
 z,_=calc(h);print('decay',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
