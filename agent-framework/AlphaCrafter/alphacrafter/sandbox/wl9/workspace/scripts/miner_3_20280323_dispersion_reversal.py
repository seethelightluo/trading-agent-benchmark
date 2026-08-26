import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-03-22'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); ret=px.pct_change(); raw=px.pct_change(3); disp=raw.std(axis=1)
# Cross-sectional 3d reversal relative to the daily cross-section, active in elevated dispersion regimes.
threshold=disp.rolling(60,min_periods=30).median(); fac=-(raw.sub(raw.mean(axis=1),axis=0)).where(disp.gt(threshold),np.nan)
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt);ns.append(len(z))
 a=np.asarray(vals); di=pd.DatetimeIndex(dates)
 def m(x): return len(x),float(np.mean(x)),float(np.mean(x)/(np.std(x,ddof=1)/np.sqrt(len(x)))),float(np.mean(x>0))
 print('horizon',h,'all',m(a),'online',m(a[di>=pd.Timestamp('2026-07-16')]),'recent',m(a[di>=pd.Timestamp('2027-03-23')]),'mean_n',float(np.mean(ns)))
print('coverage',float(fac.notna().mean().mean()),'turnover',float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),'period',px.index.min().date(),px.index.max().date())
