import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-05-31')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index()
r=px.pct_change()
# Directional path efficiency: signed trailing return divided by total absolute path movement.
fac=(px/px.shift(20)-1)/(r.abs().rolling(20,min_periods=15).sum()+1e-8)
def stat(x):
 x=np.asarray(x); return len(x),float(np.nanmean(x)),float(np.nanmean(x)/(np.nanstd(x,ddof=1)/np.sqrt(len(x)))),float(np.mean(x>0))
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.array(vals); d=pd.to_datetime(dates)
 print(h,'d',stat(a),'avgN',round(np.mean(ns),2),'online',stat(a[d>=pd.Timestamp('2026-07-16')]),'recent252',stat(a[-252:]))
print('coverage',float(fac.notna().mean().mean()),'dates',len(fac))
rank=fac.rank(axis=1,pct=True); print('turnover',float(rank.diff().abs().mean(axis=1).mean()*2))
