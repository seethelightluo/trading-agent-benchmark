import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=None
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:d=fn(s,days=4000)
  except: d=None
  if d is not None and len(d)>100:break
 if d is not None and len(d)>100:
  z=d.copy();z.date=pd.to_datetime(z.date);P[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff()
# location in rolling 60d range, fade extremes; normalized by recent volatility
lo=p.rolling(60,min_periods=40).min(); hi=p.rolling(60,min_periods=40).max()
loc=(p-lo)/(hi-lo)-.5
f=-loc/(r.rolling(20,min_periods=15).std()*np.sqrt(20))
for h in [1,3,5,10]:
 vals=[];ns=[]; fr=f.shift(1); fw=r.shift(-h).rolling(h).sum().shift(-(h-1))
 for dt in f.index:
  q=pd.concat([fr.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=np.array(vals);print('H',h,'dates',len(x),'avgN',np.mean(ns),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(ns)/15)
print('instruments',len(P),'dates',len(p),'last',p.index[-1])
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20300207_rangeloc_signal.csv',index=False)
