import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-11'); b=Path('../persistent/stock_data')
px={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change()
# Drawdown recovery efficiency: recovery from 30d trough, scaled by trailing 20d volatility.
# Positive values identify assets rebounding strongly from a recent trough per unit risk.
trough=P.rolling(30,min_periods=15).min(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(P/trough-1).div(vol.replace(0,np.nan),axis=0)
y=P.shift(-10)/P-1
ics=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
a=np.asarray(ics)
print('factor=drawdown_recovery_efficiency horizon=10 end=',end.date())
print('dates',len(a),'avg_N',round(float(np.mean(ns)),3),'IC',round(float(a.mean()),6),'ICIR',round(float(a.mean()/a.std(ddof=1)),6),'hit',round(float(np.mean(a>0)),4))
print('coverage',round(float(f.notna().sum(axis=1).ge(8).mean()),4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),4))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-11')]:
 q=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.asarray(q); print('regime',lo,'dates',len(q),'IC',round(float(q.mean()),6) if len(q) else None,'ICIR',round(float(q.mean()/q.std(ddof=1)),6) if len(q)>1 else None)
f.to_csv('scripts/miner_3_20280111_drawdown_recovery_efficiency_signal.csv')
