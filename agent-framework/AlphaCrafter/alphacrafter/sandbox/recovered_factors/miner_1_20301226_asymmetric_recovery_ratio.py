import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A},axis=1).sort_index()
r=np.log(P/P.shift(1)); x=r.shift(1)
# asymmetric recovery: positive-return intensity divided by downside deviation, favoring steady upside
up=x.clip(lower=0).rolling(20,min_periods=15).mean(); dn=(-x.clip(upper=0)).rolling(20,min_periods=15).mean(); sig=up/(dn+1e-6)
print('candidate asymmetric_recovery_ratio_20; dates',len(P),'assets',len(A),'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
 y=np.log(P.shift(-h)/P); vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 v=np.array(vals);print('H',h,'dates',len(v),'meanN',np.mean(ns),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
for name,mask in [('2020-23',P.index<'2024'),('2024-27',(P.index>='2024')&(P.index<'2028')),('2028+',P.index>='2028'),('latest120',P.index>=P.index[-120])]:
 y=np.log(P.shift(-1)/P);v=[]
 for dt in P.index[mask]:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.array(v);print(name,len(v),v.mean(),v.mean()/v.std(ddof=1))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
