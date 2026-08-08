import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
END=pd.Timestamp('2031-06-25')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a,col):
 p='../persistent/stock_data/'+a+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+a+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index();return pd.to_numeric(d[col],errors='coerce')
P=pd.DataFrame({a:load(a,'close') for a in A}).loc[:END]
O=pd.DataFrame({a:load(a,'open') for a in A}).reindex(P.index); H=pd.DataFrame({a:load(a,'high') for a in A}).reindex(P.index);L=pd.DataFrame({a:load(a,'low') for a in A}).reindex(P.index)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# Candidate: stress absorption, intraday close location averaged 10d, residualized cross-sectionally on recent momentum.
stress=(v>v.rolling(120,min_periods=60).quantile(.70))&(v.diff(5)>0)&(dxy.pct_change(5)<0)
rng=(H-L).replace(0,np.nan); loc=((P-O)/rng).rolling(10,min_periods=8).mean()
mom=P.pct_change(20)
F=loc.copy()*np.nan
for dt in P.index:
 z=pd.concat([loc.loc[dt].rename('x'),mom.loc[dt].rename('m')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.m]; F.loc[dt,z.index]=z.x-X@np.linalg.lstsq(X,z.x,rcond=None)[0]
F=F.sub(F.median(axis=1),axis=0).where(stress)
print('dates',len(P),'assets',len(A),'stress_dates',int(stress.sum()),'coverage',round(F.notna().mean().mean(),4),'active_meanN',round(F.count(axis=1).where(stress).mean(),2))
for h in [1,5,10,20]:
 R=P.pct_change(h).shift(-h); q=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(q);print('H',h,'eligible',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031')]:
 R=P.pct_change(20).shift(-20);q=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(q,columns=['d','ic']).set_index('d').loc[lo:hi].ic
 print('REG',lo, len(q),round(q.mean(),6) if len(q) else None,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
# stability recent
for n in [120,252,504]:
 R=P.pct_change(20).shift(-20);q=[]
 for dt in F.index[-n:]:
  z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q);print('RECENT',n,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
