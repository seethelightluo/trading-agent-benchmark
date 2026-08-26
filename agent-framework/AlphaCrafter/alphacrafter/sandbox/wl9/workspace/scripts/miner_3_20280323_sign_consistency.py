import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-03-22'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Sign-consistency trend: net fraction of positive sessions, centered, over a 15-day window.
# Unlike raw momentum, rewards persistent direction and penalizes choppy paths.
pos=r.gt(0).rolling(15,min_periods=15).mean()
fac=2*pos-1
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.asarray(vals); dates=pd.DatetimeIndex(dates)
 def met(x):
  return (len(x),float(np.nanmean(x)),float(np.nanmean(x)/(np.nanstd(x,ddof=1)/np.sqrt(len(x)))) if len(x)>1 else np.nan,float((x>0).mean()))
 print('horizon',h,'all',met(a),'online',met(a[dates>=pd.Timestamp('2026-07-16')]),'recent',met(a[dates>=pd.Timestamp('2027-03-23')]),'mean_n',float(np.mean(ns)))
print('coverage',float(fac.notna().mean().mean()),'turnover',float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
print('period',px.index.min().date(),px.index.max().date())
