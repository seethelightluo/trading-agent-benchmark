import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); R=P.pct_change()
# Volatility-shock absorption: recent return after a volatility spike, normalized by current downside risk.
# The shock indicator is asset-specific: prior 5d realized vol above its 60d 75th percentile.
rv5=R.rolling(5,min_periods=5).std(); rv60=R.rolling(60,min_periods=40).std(); shock=(rv5>rv60.rolling(60,min_periods=40).quantile(.75)).astype(float)
down=R.where(R<0).rolling(20,min_periods=12).std(); F=(R.rolling(5,min_periods=5).sum()/down.replace(0,np.nan)).where(shock>0).fillna(0).shift(1)
print('data',P.index.min().date(),P.index.max().date(),'assets',len(A),'dates',len(P),'coverage',round(F.notna().mean().mean(),4),'shock',round(shock.mean().mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; out=[];ns=[]
 for d in P.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=np.array(out); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round(np.mean(s>0),4))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-07-08')]:
 vals=[]; y=P.shift(-1)/P-1
 for d in P.loc[lo:hi].index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals); print('REG',lo,hi,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round(np.mean(q>0),4))
