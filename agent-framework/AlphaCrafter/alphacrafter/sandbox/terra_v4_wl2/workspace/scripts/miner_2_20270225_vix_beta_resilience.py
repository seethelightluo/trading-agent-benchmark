import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index()
V=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
r=P.pct_change(); vr=V.pct_change(); m=vr.rolling(60,min_periods=40).mean(); mr=r.rolling(60,min_periods=40).mean()
cov=r.mul(vr,axis=0).rolling(60,min_periods=40).mean()-mr.mul(m,axis=0); beta=cov.div(vr.rolling(60,min_periods=40).var(),axis=0); f=-beta
for h in [1,5,10]:
 vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals); print(h,'dates',len(x),'avgN',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('../persistent/factor_signals_miner_2_20270225_vix_beta_resilience.csv',index=False)
