import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:'2027-02-24']
r=P.pct_change()
# Recovery quality: medium trend penalized by distance from long peak; favors steady recovery rather than overextended assets
trend=r.rolling(30,min_periods=20).sum()
dd=P/P.rolling(120,min_periods=60).max()-1
sig=trend + 0.5*dd
# cross-sectional rank-normalized signal
sig=sig.rank(axis=1,pct=True)
for h in [1,3,5,10]:
 vals=[]; ns=[]; dates=[]
 f=P.shift(-h)/P-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 x=np.array(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027-12-31')]:
  q=x[[str(d)[:4]>=lo and str(d)[:10]<=hi for d in dates]]
  print('REG',lo,len(q),round(q.mean(),6) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
print('coverage',round(sig.notna().sum().sum()/(len(U)*len(sig)),4),'turnover',round(sig.rank(pct=True).diff().abs().mean().mean(),4))
# artifact in case passes
out=sig.stack().rename('signal').reset_index();out.columns=['date','asset','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_recovery_quality.csv',index=False)
