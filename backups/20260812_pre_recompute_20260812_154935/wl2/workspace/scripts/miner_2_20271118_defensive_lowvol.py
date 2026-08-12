import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s,idx):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); return d.set_index(d.date.dt.normalize()).close.reindex(idx)
idx=pd.DatetimeIndex(pd.read_csv('../persistent/stock_data/SPX.csv',parse_dates=['date']).date.dt.normalize().unique()); idx=idx[(idx>='2020-04-01')&(idx<='2027-11-17')]
C=pd.DataFrame({s:ld(s,idx) for s in U}); R=C.pct_change(); v=R.rolling(20,min_periods=15).std(); m=R.mean(axis=1).rolling(20,min_periods=15).std()
# Prefer low idiosyncratic risk, but reward defensive assets in high common-volatility states.
F=(-v).where(m.shift(1)<m.shift(1).rolling(120,min_periods=60).median(),(-v*1.5)).shift(1)
F=F.sub(F.median(axis=1),axis=0)
print('dates',len(idx),'instruments',len(U),'stress share',round((m.shift(1)>=m.shift(1).rolling(120,min_periods=60).median()).mean(),4))
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; z=[];ns=[]
 for d in idx:
  q=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
 z=np.asarray(z);print('h',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(pct=True).diff().abs().mean().mean(),6))
print('last_date',idx[-1].date())
