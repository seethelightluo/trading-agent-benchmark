import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
R={}
for a in A:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); R[a]=d.close.pct_change()
r=pd.DataFrame(R); m=r.mean(axis=1); # residual to contemporaneous cross-sectional mean, all lagged
# 8-day residual reversal, volatility scaled, with 3-day return confirmation penalty
res=r.sub(m,axis=0)
vol=r.rolling(40,min_periods=20).std()*np.sqrt(8)
F=(-(res.rolling(8,min_periods=6).sum())/(vol+1e-9)).shift(1)
P=pd.DataFrame({a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date').close for a in R})
for h in [5,10,20,30]:
 vals=[]; ns=[]
 for dt in F.index:
  if dt not in P.index: continue
  z=pd.concat([F.loc[dt],(P.shift(-h).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(vals); print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
print('coverage',F.notna().sum(axis=1).mean()/len(A),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
F.to_csv('scripts/artifacts/miner_3_20330317_short_residual_reversal_signal.csv')
