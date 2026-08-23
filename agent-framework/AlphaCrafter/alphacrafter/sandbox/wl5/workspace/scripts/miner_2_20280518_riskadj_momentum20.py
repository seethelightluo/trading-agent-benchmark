import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s, days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); ret=p.pct_change()
f=ret.rolling(20,min_periods=15).sum()/ret.rolling(20,min_periods=15).std(); fr=p.pct_change().shift(-1)
ics=[]; dates=[]; nobs=[]
for dt in f.index:
 x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(x)>=8:
  c=x.iloc[:,0].corr(x.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);dates.append(dt);nobs.append(len(x))
ics=np.array(ics); print('dates',len(ics),'avg_n',np.mean(nobs),'coverage',np.mean(nobs)/len(U))
icir=np.mean(ics)/np.std(ics,ddof=1)
to=np.nanmean(np.abs(f.diff()).sum(axis=1)/f.notna().sum(axis=1))
print('IC %.8f ICIR %.8f hit %.5f turnover %.6f'%(np.mean(ics),icir,np.mean(ics>0),to))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-05-18')]:
 z=ics[(np.array(dates)>=pd.Timestamp(a))&(np.array(dates)<=pd.Timestamp(b))]; print(a,b,'n',len(z),'IC',np.mean(z) if len(z) else np.nan,'ICIR',np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
f.to_csv('scripts/miner_2_20280518_riskadj_momentum20_signal.csv')
