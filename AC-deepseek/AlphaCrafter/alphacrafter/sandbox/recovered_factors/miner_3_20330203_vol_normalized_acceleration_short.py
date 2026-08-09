import pandas as pd,numpy as np
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in watch}).sort_index().loc[:'2033-01-18']
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Short acceleration: latest 5d return minus preceding 5d return, volatility scaled, lagged one day.
sig=((px.pct_change(5)-px.pct_change(5).shift(5))/(vol*np.sqrt(10)+1e-10)).shift(1)
print('candidate=vol_normalized_acceleration_5_5; dates',len(px),'assets',len(watch),'coverage',round(sig.notna().mean().mean(),4))
for h in [1,5,10,20]:
 fwd=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2033-01-18')]:
 vals=[];fwd=px.pct_change(10).shift(-10)
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print('REG10',lo,hi,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else np.nan)
print('turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'meanvalid',round(sig.notna().sum(axis=1).replace(0,np.nan).mean(),2))
print('AUDIT_REQUIRED: exact library signal reconstruction pending; do not persist without max_abs_library_correlation')
