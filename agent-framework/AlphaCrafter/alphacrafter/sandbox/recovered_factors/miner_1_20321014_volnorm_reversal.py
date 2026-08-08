import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index().loc[:'2032-10-13']; R=P.pct_change()
# Volatility-normalized short reversal: reverse 5d return after dividing by each asset's 20d vol;
# intended to avoid letting crypto-sized moves dominate cross-sectional ranks.
vol=R.rolling(20,min_periods=15).std(); F=(-R.rolling(5,min_periods=5).sum()/vol).shift(1)
print('candidate vol_normalized_5d_reversal; dates',len(P),'assets',len(A),'coverage',round(F.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; out=[];ns=[]
 for d in P.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=np.array(out); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round(np.mean(s>0),4))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-10-13')]:
 y=P.shift(-10)/P-1; v=[]
 for d in P.loc[lo:hi].index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(v);print('REG10',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
