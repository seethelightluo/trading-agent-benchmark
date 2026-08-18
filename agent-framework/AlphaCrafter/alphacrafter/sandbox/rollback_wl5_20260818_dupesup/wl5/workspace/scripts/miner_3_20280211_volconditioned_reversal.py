import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-10'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}; P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change(); r5=P.pct_change(5); v=R.rolling(20,min_periods=15).std(); raw=-(r5.sub(r5.median(axis=1),axis=0));
# volatility-conditioned relative reversal: retain contrarian signal in high cross-sectional volatility names
f=raw.where(v.ge(v.median(axis=1),axis=0))
y=P.shift(-10)/P-1;a=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
a=np.array(a);print('dates',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0));print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-10')]:
 q=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
f.to_csv('scripts/miner_3_20280211_volconditioned_reversal_signal.csv')
