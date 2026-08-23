import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-11-29')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.DataFrame(p).sort_index().loc[:cut]; r=p.pct_change()
# Volatility-expansion trend: lagged 10d momentum, scaled by recent/long volatility.
# All inputs are shifted one day to avoid look-ahead.
f=(p.shift(1)/p.shift(11)-1)*(r.shift(1).rolling(5,min_periods=5).std()/(r.shift(1).rolling(40,min_periods=30).std()+1e-8))
f=f.replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; a=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(a); print('h',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(np.array(ns)/15),4))
print('turnover',np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=np.asarray([v for d,v in zip(ds,a) if lo<=d.year<=hi]); print('regime',lo,hi,'dates',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
