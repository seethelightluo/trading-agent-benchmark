import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date')
    px[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index()
# Candidate: lagged short-horizon reversal, normalized by trailing volatility.
# Signal at t uses close through t; forward return begins t+1.
ret=prices.pct_change()
vol=ret.rolling(20,min_periods=15).std()
sig=-(prices/prices.shift(5)-1)/(vol*np.sqrt(5)+0.02)
# winsorize cross-sectionally each day
sig=sig.clip(lower=sig.quantile(.05,axis=1),upper=sig.quantile(.95,axis=1),axis=0)
rows=[]
for i in range(len(prices)-1):
    a=sig.iloc[i]; f=prices.iloc[i+1]/prices.iloc[i]-1
    z=pd.concat([a,f],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((prices.index[i],ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=5d_vol_scaled_reversal')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
for label, x in [('all',r),('2026+',r[r.index>='2026-01-01']),('2030+',r[r.index>='2030-01-01']),('2032+',r[r.index>='2032-01-01'])]:
    print(label,'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()),'dates',len(x))
# rank turnover
rank=sig.rank(axis=1,pct=True)
turn=(rank.diff().abs().mean(axis=1)).dropna().mean()
print('turnover',turn)
# decay at 5,10,20 sessions
for h in [5,10,20]:
  vals=[]
  for i in range(len(prices)-h):
    z=pd.concat([sig.iloc[i],prices.iloc[i+h]/prices.iloc[i]-1],axis=1).dropna()
    if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  x=pd.Series(vals); print('h',h,'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std()),'dates',len(x))
