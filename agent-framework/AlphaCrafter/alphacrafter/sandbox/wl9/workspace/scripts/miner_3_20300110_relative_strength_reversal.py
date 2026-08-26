import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2030-01-10')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}; p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change(); r10=p.pct_change(10)
# Residual reversal: lagged asset return relative to cross-asset median, favor underperformers.
f=-(r10.sub(r10.median(axis=1),axis=0)); sig=f.shift(1)
for h in [5,10,20,40]:
 y=p.shift(-h)/p-1; a=[]; ns=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a); print(h,len(a),round(np.mean(ns),2),round(np.mean(ns)/15,4),round(a.mean(),6),round(a.mean()/(a.std(ddof=1)+1e-12),6),round(np.mean(a>0),4))
out=f;out.index.name='date';out.to_csv('scripts/miner_3_20300110_relative_strength_reversal_10d_signal.csv');print('range',p.index.min(),p.index.max())
