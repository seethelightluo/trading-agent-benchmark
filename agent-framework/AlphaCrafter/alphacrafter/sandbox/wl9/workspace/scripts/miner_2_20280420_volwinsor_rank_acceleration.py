import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-04-19')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index()
r=px.pct_change(); short=px/px.shift(5)-1; long=px/px.shift(20)-1
acc=short.rank(axis=1,pct=True)-long.rank(axis=1,pct=True)
vol=r.rolling(20,min_periods=15).std()
# Winsorize asset volatility to each date's 10th/90th percentiles before scaling.
lo=vol.quantile(.10,axis=1); hi=vol.quantile(.90,axis=1)
vw=vol.clip(lower=lo,upper=hi,axis=0)
fac=acc/vw.replace(0,np.nan)
records=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],px.shift(-5).loc[dt]/px.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: records.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=np.array([x[2] for x in records]); d=pd.to_datetime([x[0] for x in records]); ns=np.array([x[1] for x in records])
def stat(x):
 return (len(x),float(np.mean(x)),float(np.mean(x)/(np.std(x,ddof=1)/np.sqrt(len(x)))) if len(x)>1 else np.nan,float(np.mean(x>0)))
print('period',px.index.min().date(),end.date(),'dates',len(a),'avgN',round(ns.mean(),2),'coverage',round(float(fac.notna().mean().mean()),4))
print('5d',stat(a))
for label,m in [('online',d>=pd.Timestamp('2026-07-16')),('2027+',d>=pd.Timestamp('2027-01-01')),('2028+',d>=pd.Timestamp('2028-01-01')),('recent252',np.arange(len(a))>=len(a)-252)]: print(label,stat(a[m]))
for h in [1,10,20]:
 q=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],px.shift(-h).loc[dt]/px.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(str(h)+'d',stat(np.array(q)))
rank=fac.rank(axis=1,pct=True)
print('turnover',float(rank.diff().abs().mean(axis=1).mean()*2))
