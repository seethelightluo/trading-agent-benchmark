import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-10'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change()
r5=P.pct_change(5); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
# Relative 5d reversal, scaled by trailing volatility; larger means stronger contrarian signal
f=-(r5.sub(r5.median(axis=1),axis=0)).div(vol.replace(0,np.nan))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(a); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
rk=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round((rk-rk.shift()).abs().mean(axis=1).dropna().mean(),4))
y=P.shift(-10)/P-1
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-10')]:
 a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(a); print('REG',lo,hi,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
print('period',P.index.min().date(),P.index.max().date())
# artifact for deterministic post-gate provenance
f.to_csv('scripts/miner_3_20280211_volscaled_relative_reversal_5d10d_signal.csv')
