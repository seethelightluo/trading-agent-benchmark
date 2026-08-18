import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-05-22'); base=Path('../persistent/stock_data')
d={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:x.close for s,x in d.items()}).sort_index().loc[:end].ffill(); r=P.pct_change()
# Trend acceleration: recent 20d return minus preceding 40d return, normalized by trailing 20d volatility.
f=((P/P.shift(20)-1)-(P.shift(20)/P.shift(60)-1))/(r.rolling(20).std()*np.sqrt(20)); y=P.shift(-10)/P-1
A=[]; ns=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): A.append(q); ns.append(len(z))
a=np.asarray(A)
print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-05-22')]:
 b=[]
 for dt in P.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): b.append(q)
 b=np.asarray(b); print('regime',lo,hi,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1),'hit',np.mean(b>0))
rk=f.rank(axis=1,pct=True); print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',((rk-rk.shift()).abs().mean(axis=1).dropna().mean()),'avgN',f.notna().sum(axis=1).mean())
f.to_csv('scripts/miner_1_20280523_trend_acceleration_signal.csv')
