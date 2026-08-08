import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
R=P.pct_change()
# Downside-adjusted medium-term momentum: reward 30d return per downside deviation,
# then remove cross-sectional level to emphasize risk-efficient leadership.
down=R.clip(upper=0).pow(2).rolling(30,min_periods=20).mean().pow(.5)
raw=R.rolling(30,min_periods=20).sum()/down.replace(0,np.nan)
# orthogonalize to ordinary 20d trend, preserving interpretable risk-adjustment
mom=R.rolling(20,min_periods=15).sum()
F=raw.copy()
for dt in P.index:
 z=pd.concat([raw.loc[dt],mom.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,1].std()>0:
  x=z.iloc[:,1].values;y=z.iloc[:,0].values
  b=np.cov(x,y,ddof=1)[0,1]/np.var(x,ddof=1)
  F.loc[dt,z.index]=y-b*x
F=F.shift(1)
print('idea=orthogonal_downside_adjusted_momentum_30 data',P.index.min().date(),P.index.max().date(),'assets',len(A),'dates',len(P),'coverage',round(F.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; out=[]; ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=np.array(out);print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round(np.mean(s>0),4))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-08-18')]:
 vals=[]
 for dt in P.loc[lo:hi].index:
  z=pd.concat([F.loc[dt],(P.shift(-1)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals);print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
# decay via horizons above
