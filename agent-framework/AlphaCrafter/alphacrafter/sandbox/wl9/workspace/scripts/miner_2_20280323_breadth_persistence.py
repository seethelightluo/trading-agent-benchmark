import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-03-22')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index()
ret=px.pct_change(); med=ret.median(axis=1)
# breadth persistence: fraction of last 20 sessions outperforming cross-sectional median, centered
fac=(ret.gt(med,axis=0).rolling(20,min_periods=15).mean()-0.5)
fwd=px.shift(-10)/px-1;a=[];ds=[];ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
a=np.array(a);d=pd.to_datetime(ds)
def stat(x): return (len(x),x.mean(),x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan,(x>0).mean())
print('dates',len(a),'avgN',np.mean(ns),'coverage',fac.notna().mean().mean(),'IC/ICIR/hit',stat(a))
for lab,m in [('recent252',np.arange(len(a))>=len(a)-252),('2026+',d>=pd.Timestamp('2026-07-16')),('2027+',d>=pd.Timestamp('2027-01-01')),('2028+',d>=pd.Timestamp('2028-01-01'))]: print(lab,stat(a[m]))
for h in [1,5,20]:
 f=px.shift(-h)/px-1; q=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,stat(np.array(q)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2)
