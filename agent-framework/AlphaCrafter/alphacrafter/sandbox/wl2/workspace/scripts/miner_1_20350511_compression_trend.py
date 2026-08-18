import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is not None: px[s]=d.set_index('date')['close']
C=pd.DataFrame(px).sort_index().ffill(); R=C.pct_change()
# Candidate: medium-term trend normalized by recent volatility, activated only in
# compressed cross-asset dispersion. All signal inputs are lagged one session.
trend=R.rolling(30,min_periods=25).sum()
vol=R.rolling(20,min_periods=15).std()
disp=R.rolling(10,min_periods=8).std().mean(axis=1)
compression=disp < disp.rolling(120,min_periods=60).median()
S=(trend/vol).shift(1).where(compression.shift(1),0.0)
print('rows=%d assets=%d dates=%d'%(len(C),len(px),len(S)))
for h in [5,10,20,40]:
 f=C.shift(-h)/C-1; vals=[]; ns=[]
 for dt in S.index:
  z=pd.concat([S.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 q=pd.Series(vals).dropna(); print('h%d dates=%d avgN=%.2f IC=%.8f ICIR=%.8f hit=%.4f'%(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean()))
print('coverage=%.4f activation=%.4f'%(S.notna().sum(axis=1).mean()/len(U),compression.mean()))
S.index.name='date'; S.to_csv('../persistent/miner_1_20350511_compression_trend_signal.csv')
