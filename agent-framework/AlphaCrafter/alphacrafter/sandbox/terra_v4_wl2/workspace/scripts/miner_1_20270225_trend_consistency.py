import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'; ds={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date'); ds[s]=d.close.astype(float)
px=pd.concat(ds,axis=1).sort_index(); rets=px.pct_change()
# trend consistency: signed fraction of positive daily returns, centered, times magnitude of 20d return
cons=rets.rolling(20).mean()/rets.rolling(20).std() # interpretable consistency / noise
# alternative only one idea: consistency of direction
fac=rets.gt(0).rolling(20).mean()-0.5
# use direction consistency; forward returns
for h in [1,3,5,10]:
 fwd=px.shift(-h)/px-1; rows=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=np.array([x[1] for x in rows]); dates=pd.to_datetime([x[0] for x in rows])
 print('H',h,'dates',len(a),'avgN',np.mean([x[2] for x in rows]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2026,2027)]:
  q=a[(dates.year>=lo)&(dates.year<=hi)]; print(' regime',lo,hi,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('coverage',fac.notna().sum().sum()/(fac.shape[0]*fac.shape[1]))
# artifact long format
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_trend_consistency.csv',index=False)
