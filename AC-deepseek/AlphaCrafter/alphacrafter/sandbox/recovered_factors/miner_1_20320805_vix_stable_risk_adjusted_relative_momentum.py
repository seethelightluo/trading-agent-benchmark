import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); R=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# VIX-regime conditional risk-adjusted relative momentum: rank-centered 20d return / 20d vol, activated only on stable/falling VIX.
ret20=P.pct_change(20); vol20=R.rolling(20,min_periods=15).std(); raw=ret20/vol20.replace(0,np.nan); rel=raw.sub(raw.median(axis=1),axis=0)
gate=(v.pct_change(5)<=0).astype(float).replace(0,np.nan); F=rel.mul(gate,axis=0).shift(1)
print('idea=vix_stable_risk_adjusted_relative_momentum',P.index.min().date(),P.index.max().date(),'assets',len(A),'dates',len(P),'coverage',round(F.notna().mean().mean(),4),'gate',round(gate.notna().mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; out=[];ns=[]
 for d in P.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=np.array(out); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round(np.mean(s>0),4))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
y=P.shift(-1)/P-1
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-07-15')]:
 vals=[]
 for d in P.loc[lo:hi].index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals); print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round(np.mean(q>0),4) if len(q) else None)
