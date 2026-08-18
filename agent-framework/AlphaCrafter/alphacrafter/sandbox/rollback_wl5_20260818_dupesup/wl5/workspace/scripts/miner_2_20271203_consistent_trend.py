import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-12-03'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change()
# Consistency-adjusted trend: return times directional consistency, avoiding raw momentum dominance
ret=R.rolling(20,min_periods=15).sum(); consistency=(R.gt(0).rolling(20,min_periods=15).mean()-0.5)*2
f=ret*consistency
sig=f.rename_axis('date').stack().rename('signal').reset_index(); sig.to_csv('scripts/miner_2_20271203_consistent_trend_signal.csv',index=False)
print('signal_artifact',len(sig))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; ic=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(ic); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
rk=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round((rk-rk.shift()).abs().mean(axis=1).dropna().mean(),4))
y=P.shift(-20)/P-1
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-03')]:
 a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(a); print('REG',label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
print('period',P.index.min().date(),P.index.max().date(),'instruments',len(U))
