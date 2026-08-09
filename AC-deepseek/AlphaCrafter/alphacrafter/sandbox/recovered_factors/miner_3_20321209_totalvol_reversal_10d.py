import pandas as pd,numpy as np,glob,json,os
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in watch}).sort_index()
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()
# candidate: total-vol scaled 10d reversal, lagged one day
sig=(-(px.pct_change(10))/vol).shift(1)
fwd={h:px.pct_change(h).shift(-h) for h in [1,5,10,20]}
print('rows',len(px),'assets',len(watch),'cells',sig.notna().sum().sum(),'coverage',sig.notna().mean().mean())
for h in fwd:
 vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('turn10',np.nanmean(sig.rank(axis=1,pct=True).diff(10).abs().mean(axis=1)))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 vals=[]
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fwd[10].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals);print('REG',lo,hi,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
# library audit: reconstruct admitted expressions approximately by loading factor scripts impossible; compare candidate to all stored factor signal-like? report available factor files and omit if no evidence
print('AUDIT_NOTE library signals require exact reconstruction; no persistence unless completed')
