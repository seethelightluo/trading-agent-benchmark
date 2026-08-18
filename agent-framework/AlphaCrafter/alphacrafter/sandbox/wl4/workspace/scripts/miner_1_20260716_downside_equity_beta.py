import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); return d.close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index().ffill(); r=np.log(p).diff()
# Downside beta to global equity basket: covariance only on basket-negative sessions.
eq=r[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
neg=eq.where(eq<0)
fac=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 fac[s]=-r[s].rolling(60,min_periods=30).cov(neg)/neg.rolling(60,min_periods=30).var()
ics=[]; ns=[]; turns=[]; prev=None; dates=[]
for i in range(60,len(p)-1):
 q=pd.concat([fac.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
 if len(q)>=8:
  ics.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(p.index[i])
  if prev is not None: turns.append(np.mean(np.sign(fac.iloc[i])!=np.sign(prev)))
  prev=fac.iloc[i]
a=np.asarray(ics); print('dates',len(a),'avgN',np.mean(ns),'coverage',np.sum(ns)/(len(ns)*15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(turns))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 b=a[[lo<=str(x.date())[:4]<=hi for x in dates]];print(lo,hi,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1))
for h in [5,10]:
 z=[]
 for i in range(60,len(p)-h):
  q=pd.concat([fac.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print('h',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
