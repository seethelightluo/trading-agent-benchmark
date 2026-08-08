import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index();R=P.pct_change()
# Relative leadership persistence: asset 40d return minus contemporaneous cross-asset median,
# scaled by its 20d realized volatility; removes common market direction and risk-loadings.
rel=R.rolling(40,min_periods=30).sum().sub(R.rolling(40,min_periods=30).sum().median(axis=1),axis=0)
vol=R.rolling(20,min_periods=15).std()
F=(rel/vol.replace(0,np.nan)).shift(1)
print('idea=relative_leadership_volscaled_40 data',P.index.min().date(),P.index.max().date(),'assets',len(A),'dates',len(P),'coverage',round(F.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1;out=[];ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=np.array(out);print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round(np.mean(s>0),4))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-08-18')]:
 vals=[]
 for dt in P.loc[lo:hi].index:
  z=pd.concat([F.loc[dt],(P.shift(-1)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals);print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
