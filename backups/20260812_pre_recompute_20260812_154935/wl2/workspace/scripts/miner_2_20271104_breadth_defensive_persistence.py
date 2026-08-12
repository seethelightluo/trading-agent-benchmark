import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,idx):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); d.date=d.date.dt.normalize(); return d.set_index('date').close.reindex(idx)
base=pd.read_csv('../persistent/stock_data/SPX.csv',parse_dates=['date']); idx=pd.DatetimeIndex(base.date.dt.normalize().unique());idx=idx[(idx>='2020-04-01')&(idx<='2027-11-03')]
C=pd.DataFrame({s:load(s,idx) for s in U}); R=C.pct_change(); vol=R.rolling(20,min_periods=15).std()
r5=R.rolling(5,min_periods=5).sum(); r10=R.rolling(10,min_periods=10).sum(); breadth=(r5>0).mean(axis=1)
raw=r10/vol; rev=-r5/vol; F=raw.where(breadth>=.5,rev).shift(1); F=F.sub(F.median(axis=1),axis=0)
print('dates',len(idx),'instruments',len(U),'breadth bear share',float((breadth<.5).mean()))
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; z=[];ns=[]
 for d in idx:
  q=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
 z=np.asarray(z); print('h',h,'dates',len(z),'avgN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(pct=True).diff().abs().mean().mean())
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 z=[]
 for d in idx[(idx.year>=int(a))&(idx.year<=int(b))]:
  q=pd.concat([F.loc[d].rename('f'),(C.shift(-1)/C-1).loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.f,q.y).statistic)
 z=np.asarray(z);print(a+'-'+b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
