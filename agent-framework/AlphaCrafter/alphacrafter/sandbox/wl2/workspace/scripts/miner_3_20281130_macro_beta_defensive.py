import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=5000)
 return d
D={s:fetch(s) for s in U}; close=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill()
macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(close.index).ffill()
r=np.log(close).diff(); vm=np.log(macro).diff()
beta=r.rolling(60,min_periods=40).cov(vm).div(vm.rolling(60,min_periods=40).var(),axis=0)
shock=(macro/macro.rolling(60,min_periods=40).median()-1).clip(-2,2)
variants={'beta_shock':beta.rolling(5).mean().mul(shock,axis=0),'beta_level':beta.rolling(5).mean().mul((macro/macro.rolling(120,min_periods=60).median()).clip(.5,2),axis=0),'beta_shock_plus_mom':beta.rolling(5).mean().mul(shock,axis=0)+r.rolling(20).sum()/r.rolling(20).std()}
for name,f0 in variants.items():
 f=f0.shift(1)
 for h in [1,5,10,20]:
  ic=[]; ns=[]; fr=close.shift(-h)/close-1
  for dt in f.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8: ic.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank())); ns.append(len(z))
  a=np.array(ic); print(name,h,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6),round(np.mean(a>0),4),round(np.mean(ns),2))
 h=10; fr=close.shift(-h)/close-1; ic=[]
 for dt in f.index[f.index>='2027-01-01']:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print(name,'recent10',len(ic),round(np.nanmean(ic),6),round(np.nanmean(ic)/np.nanstd(ic,ddof=1),6))
print('range',close.index.min(),close.index.max(),'assets',close.notna().sum(axis=1).median())
f=variants['beta_shock_plus_mom'].shift(1); f.index.name='date'; f.to_csv('scripts/miner_3_20281130_macro_beta_shock_momentum_signal.csv')
