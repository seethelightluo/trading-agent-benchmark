import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=5000)
 return d
D={s:fetch(s) for s in U}
close=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill()
r=np.log(close).diff()
disp=r.T.rolling(5).std().T.mean(axis=1)
disp_z=(disp-disp.rolling(120,min_periods=60).mean())/disp.rolling(120,min_periods=60).std()
rev=-r.rolling(5).sum()/r.rolling(20).std()
f=rev.mul((1+disp_z.clip(-1,2)),axis=0).shift(1)
print('range',close.index.min().date(),close.index.max().date(),'assets',close.notna().sum(axis=1).median())
for h in [1,5,10,20]:
 fr=close.shift(-h)/close-1; ic=[]; ns=[]; turns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank())); ns.append(len(z))
  prev=f.shift(1).loc[dt]; both=pd.concat([f.loc[dt],prev],axis=1).dropna()
  if len(both)>=8: turns.append(both.iloc[:,0].rank().corr(both.iloc[:,1].rank()))
 a=np.array(ic)
 print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4),'turnover',round(1-np.nanmean(turns),4))
for start in ['2020-01-01','2023-01-01','2025-01-01','2027-01-01']:
 fr=close.shift(-10)/close-1; a=[]
 for dt in f.index[f.index>=start]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 a=np.array(a); print('regime',start,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6))
f.index.name='date'; f.to_csv('scripts/miner_3_20281214_dispersion_reversal_signal.csv')
