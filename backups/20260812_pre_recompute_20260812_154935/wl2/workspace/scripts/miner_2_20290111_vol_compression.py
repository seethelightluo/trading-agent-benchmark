import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,days=5000)
 return d
D={s:fetch(s) for s in U}; close=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill(); r=np.log(close).diff()
# Volatility compression: recent volatility relative to long volatility, inverse so compressed assets rank high.
v5=r.rolling(5).std(); v60=r.rolling(60).std(); f=-(v5/v60).replace([np.inf,-np.inf],np.nan).shift(1)
print('range',close.index.min().date(),close.index.max().date(),'assets',close.notna().sum(axis=1).median())
for h in [1,5,10,20]:
 fr=close.shift(-h)/close-1; a=[]; ns=[]; ts=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ns.append(len(z))
  q=pd.concat([f.loc[dt],f.shift(1).loc[dt]],axis=1).dropna()
  if len(q)>=8:ts.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 a=np.array(a);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(1-np.nanmean(ts),4))
fr=close.shift(-10)/close-1
for start in ['2020-01-01','2023-01-01','2025-01-01','2027-01-01']:
 a=[]
 for dt in f.index[f.index>=start]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 a=np.array(a);print('regime',start,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6))
f.index.name='date';f.to_csv('scripts/miner_2_20290111_vol_compression_signal.csv')
