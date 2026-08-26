import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-03-08'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Compression-continuation: medium-term directional return, strengthened when recent realized vol is compressed versus its medium baseline.
vol5=r.rolling(5,min_periods=5).std(); vol20=r.rolling(20,min_periods=20).std()
compression=(vol5/vol20).clip(0.25,2.5)
fac=px.pct_change(10) * (1.5-compression)
# cross-sectional rank IC, with explicit regime split and online split
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; n=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); n.append(len(z))
 a=np.asarray(vals); dates=pd.DatetimeIndex(dates)
 def met(x): return (len(x),float(np.nanmean(x)),float(np.nanmean(x)/(np.nanstd(x,ddof=1)/np.sqrt(len(x)))) if len(x)>1 else np.nan,float((x>0).mean()))
 print('horizon',h,'all',met(a),'online',met(a[dates>=pd.Timestamp('2026-07-16')]),'recent',met(a[dates>=pd.Timestamp('2027-03-09')]),'mean_n',np.mean(n))
print('coverage',float(fac.notna().mean().mean()),'turnover',float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
print('period',px.index.min().date(),px.index.max().date())
