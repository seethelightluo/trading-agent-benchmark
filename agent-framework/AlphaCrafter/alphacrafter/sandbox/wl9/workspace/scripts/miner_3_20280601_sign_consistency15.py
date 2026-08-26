import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-05-31'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fac=2*r.gt(0).rolling(15,min_periods=15).mean()-1
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.asarray(vals); dates=pd.DatetimeIndex(dates); print('horizon',h,'dates',len(a),'mean_n',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round((a>0).mean(),4))
 for name,cut in [('online','2026-07-16'),('recent','2027-05-31'),('ytd','2028-01-01')]:
  q=a[dates>=pd.Timestamp(cut)]; print(name,'dates',len(q),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6),'hit',round((q>0).mean(),4))
print('period',px.index.min().date(),end.date(),'instruments',len(S),'coverage',round(fac.notna().mean().mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
