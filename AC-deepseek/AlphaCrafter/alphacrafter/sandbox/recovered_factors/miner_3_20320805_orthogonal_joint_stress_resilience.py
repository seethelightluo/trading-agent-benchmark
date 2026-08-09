import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
R=P.pct_change()
def macro(n):
 x=pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
 return x.pct_change(5)
v=macro('VIX'); d=macro('DXY')
# Continuous joint stress transition: positive VIX and DXY moves define stress; resilience is return
stress=(v.rank(pct=True)+d.rank(pct=True))/2
# asset's rolling beta to stress, with returns contemporaneous; low beta = resilience
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 F[a]=R[a].rolling(60,min_periods=40).cov(stress)
# negate beta, then residualize cross-sectionally against 20d momentum to isolate stress resilience
mom=R.rolling(20).sum()
raw=-F
# cross-sectional linear residual each date
for dt in P.index:
 z=pd.concat([raw.loc[dt],mom.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,1].std()>0:
  x=z.iloc[:,1].values; y=z.iloc[:,0].values
  b=np.cov(x,y,ddof=1)[0,1]/np.var(x,ddof=1); raw.loc[dt,z.index]=y-b*x
F=raw.shift(1)
print('idea=orthogonal_continuous_joint_stress_resilience data',P.index.min().date(),P.index.max().date(),'assets',len(A),'dates',len(P),'coverage',round(F.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; out=[]; ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=np.array(out)
 print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round(np.mean(s>0),4))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-07-15')]:
 vals=[]
 for dt in P.loc[lo:hi].index:
  z=pd.concat([F.loc[dt],(P.shift(-1)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals); print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
