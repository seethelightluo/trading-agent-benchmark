import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
px=pd.DataFrame({s:d[s]['close'] for s in U}); r=px.pct_change(); m=r['SPX']
beta=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U: beta[s]=r[s].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var()
f=-beta; fr=r.shift(-1); ics=[]; nms=[]; dates=[]
for dt in f.index:
 ok=f.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8: ics.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic); nms.append(ok.sum()); dates.append(dt)
ics=np.array(ics); dates=np.array(dates,dtype='datetime64[ns]'); print('idea=negative_rolling_beta_60d dates',len(ics),'avg_names',np.mean(nms),'IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0),'coverage',np.mean(nms)/15)
for label,mask in [('2020-22',dates<=np.datetime64('2022-12-31')),('2023-24',(dates>=np.datetime64('2023-01-01'))&(dates<=np.datetime64('2024-12-31'))),('2025-26',(dates>=np.datetime64('2025-01-01'))&(dates<=np.datetime64('2026-12-31'))),('2027',dates>=np.datetime64('2027-01-01'))]:
 z=ics[mask]; print(label,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_1_20270225_lowbeta60.csv',index=False)
