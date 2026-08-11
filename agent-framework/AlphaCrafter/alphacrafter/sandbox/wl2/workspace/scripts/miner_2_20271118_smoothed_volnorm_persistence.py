import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,idx):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); d.date=d.date.dt.normalize(); return d.set_index('date').close.reindex(idx)
base=pd.read_csv('../persistent/stock_data/SPX.csv',parse_dates=['date']); idx=pd.DatetimeIndex(base.date.dt.normalize().unique()); idx=idx[(idx>='2020-04-01')&(idx<='2027-11-17')]
C=pd.DataFrame({s:load(s,idx) for s in U}); R=C.pct_change(); vol=R.rolling(20,min_periods=15).std()
raw=R.rolling(10,min_periods=10).sum()/vol
# 3-day causal EMA of the already one-day-lagged signal
F=raw.shift(1).ewm(span=3,adjust=False,min_periods=1).mean(); F=F.sub(F.median(axis=1),axis=0)
print('dates',len(idx),'instruments',len(U))
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; z=[]; ns=[]
 for d in idx:
  q=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
 z=np.asarray(z); print('h',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(pct=True).diff().abs().mean().mean(),6))
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 z=[]
 for d in idx[(idx.year>=int(a))&(idx.year<=int(b))]:
  q=pd.concat([F.loc[d].rename('f'),(C.shift(-1)/C-1).loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.f,q.y).statistic)
 z=np.asarray(z); print(a+'-'+b,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('last_date',idx[-1].date())
