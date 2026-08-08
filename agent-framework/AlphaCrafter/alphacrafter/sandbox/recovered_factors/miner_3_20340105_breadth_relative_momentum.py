import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
c=pd.DataFrame({a:D[a].close for a in assets}).sort_index(); r=np.log(c).diff()
# Relative momentum: 20d asset return relative to cross-sectional median, conditioned on market breadth
mom=r.rolling(20,min_periods=15).sum(); rel=mom.sub(mom.median(axis=1),axis=0)
breadth=(mom>0).mean(axis=1); # emphasize continuation when breadth is broad, reversal when narrow
sig=(rel*(2*abs(breadth-.5)+.25)).shift(1)
print('coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
 f=c.shift(-h)/c-1; a=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a); print(h,a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),len(a),np.mean(ns))
print('turn',np.nanmean((sig.rank(axis=1,pct=True)-sig.rank(axis=1,pct=True).shift(10)).abs().sum(axis=1)/sig.notna().sum(axis=1)))
