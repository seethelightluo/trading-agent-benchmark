import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-24'); b=Path('../persistent/stock_data'); px={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}; P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change(); r3=P.pct_change(3); disp=R.rolling(5).std().mean(axis=1); threshold=disp.rolling(60,min_periods=30).median(); raw=-(r3.sub(r3.median(axis=1),axis=0)); f=raw.where(disp.ge(threshold),np.nan); y=P.shift(-10)/P-1
a=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
a=np.array(a); print('dates',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-24')]:
 q=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print(lo,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
f.to_csv('scripts/miner_1_20280225_dispersion_conditioned_reversal_signal.csv')
