import pandas as pd, numpy as np
from scipy.stats import spearmanr
import glob, os
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}; vols={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 prices[a]=d.close; vols[a]=d.volume.replace(0,np.nan)
px=pd.DataFrame(prices); vol=pd.DataFrame(vols).reindex(px.index)
r=px.pct_change()
# Candidate: volume-weighted signed pressure over 20d, normalized by realized vol.
# positive means advances occurred on higher participation than declines.
vr=(r*vol).rolling(20,min_periods=15).sum()/vol.rolling(20,min_periods=15).sum()
sig=vr/(r.rolling(20,min_periods=15).std()*np.sqrt(20))
# Winsorize cross section not needed for rank
for h in [1,5,10,20]:
 vals=[]; dates=[]; ns=[]
 fwd=px.shift(-h)/px-1
 for dt in px.index:
  x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
# coverage turnover
print('cells',sig.notna().sum().sum(),'total',sig.size,'coverage',sig.notna().mean())
rank=sig.rank(axis=1,pct=True); print('turn10',rank.diff(10).abs().mean().mean())
for yr in [2020,2024,2028,2030]:
 mask=pd.Series(px.index.year,index=px.index)==yr
 vals=[]; fwd=px.shift(-10)/px-1
 for dt in px.index[mask]:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('year',yr,'n',len(vals),'IC',np.nanmean(vals) if vals else np.nan,'ICIR',np.nanmean(vals)/np.nanstd(vals,ddof=1) if len(vals)>1 else np.nan)
