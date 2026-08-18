import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 if d is not None:px[s]=d.set_index('date')['close']
C=pd.DataFrame(px).sort_index().ffill(); R=C.pct_change(); vol=R.rolling(40,min_periods=25).std(); raw=R.rolling(20,min_periods=20).sum()/vol; br=(R.rolling(20,min_periods=15).sum()>0).mean(axis=1); S=raw.shift(1).where(br.shift(1)>0.5,0.)
for label,idx in [('full',S.index),('recent',S.index[-750:])]:
 print(label)
 for h in [10,20]:
  f=C.shift(-h)/C-1; q=[]
  for dt in idx:
   z=pd.concat([S.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
  q=pd.Series(q);print('h%d dates=%d IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('assets',len(px),'rows',len(C),'coverage',S.notna().sum(axis=1).mean()/15)
S.index.name='date'; S.to_csv('../persistent/miner_1_20350511_breadth_trend_revalidation_signal.csv')
