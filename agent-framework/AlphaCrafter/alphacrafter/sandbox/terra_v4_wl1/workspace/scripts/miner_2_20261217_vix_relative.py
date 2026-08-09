import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
trad=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for b in ['../persistent/stock_data/','../persistent/index_data/']:
  p=b+s+'.csv'
  if os.path.exists(p):
   d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.DataFrame({s:load(s) for s in trad}).sort_index(); r=px/px.shift(5)-1
v=load('VIX'); state=(v/v.shift(10)-1>0).reindex(px.index).ffill()
fac=-(r.sub(r.median(axis=1),axis=0)).where(state, np.nan); fwd=px.shift(-1)/px-1
ics=[]; ds=[]; ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):ics.append(q);ds.append(dt);ns.append(len(z))
a=np.array(ics);print('vix rising relative reversal dates',len(a),'avg',np.mean(ns),'coverage',fac.notna().sum().sum()/fac.size,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.nanmean(fac.rank(pct=True).diff().abs().mean(axis=1)))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=a[(np.array(ds,dtype='datetime64[ns]')>=np.datetime64(lo))&(np.array(ds,dtype='datetime64[ns]')<=np.datetime64(hi))];print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
