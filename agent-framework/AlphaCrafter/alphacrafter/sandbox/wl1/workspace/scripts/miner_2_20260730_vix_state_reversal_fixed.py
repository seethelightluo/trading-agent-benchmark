import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'; P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:cut]
pd0=pd.DataFrame(P).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.loc[:cut]
st=(v>v.rolling(60,min_periods=40).median()).astype(float).reindex(pd0.index).ffill(); fac=(-(pd0/pd0.shift(5)-1)).mul(st,axis=0)
for h in [1,5,10]:
 fwd=pd0.shift(-h)/pd0-1; ics=[];ns=[];dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):ics.append(q);ns.append(len(z));dates.append(pd.Timestamp(dt))
 a=np.asarray(ics);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
   q=a[(np.array(dates,dtype='datetime64[ns]')>=np.datetime64(lo))&(np.array(dates,dtype='datetime64[ns]')<=np.datetime64(hi))];print('regime',lo,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),6));print('coverage_valid',round(fac.notna().mean().mean(),6));print('signal_artifact','VIX state aligned; factor=-5d return when VIX above trailing 60d median, zero otherwise; cutoff='+cut)
