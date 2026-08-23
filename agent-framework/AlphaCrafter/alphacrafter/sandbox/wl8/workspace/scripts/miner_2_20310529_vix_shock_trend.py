import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close=pd.DataFrame({s:get_stock_daily_data(s,3000).set_index('date')['close'].astype(float) for s in U}).sort_index()
vix=get_index_daily_data('VIX',3000).set_index('date')['close'].astype(float).sort_index()
# VIX positive shock continuation: after a sharp volatility jump, favor assets with positive 20d trend.
r20=close.pct_change(20)
vr=vix.pct_change(5)
base=vr.rolling(252,min_periods=60).rank(pct=True).reindex(close.index)
shock=((base-.8)/.2).clip(lower=0,upper=1)
sig=r20.mul(shock,axis=0)
ics=[];ns=[];tos=[];prev=None
for d in close.index:
 z=pd.concat([sig.loc[d],(close.shift(-10)/close-1).loc[d]],axis=1).dropna()
 if len(z)>=8 and sig.loc[d].abs().sum()>0:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append((d,c));ns.append(len(z))
 rank=sig.loc[d].rank(pct=True)
 if prev is not None:
  q=pd.concat([rank,prev],axis=1).dropna()
  if len(q):tos.append(float((q.iloc[:,0]-q.iloc[:,1]).abs().mean()))
 prev=rank
dt=np.array([d for d,_ in ics]); x=np.array([v for _,v in ics])
def f(a):
 a=np.asarray(a); return (float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1)),float(np.mean(a>0)),len(a)) if len(a)>1 else (np.nan,)*3+(len(a),)
print('dates',len(x),'avg_inst',np.mean(ns),'coverage',np.mean(ns)/15,'turnover',np.mean(tos))
print('10d',f(x))
for n in [180,360]: print('recent',n,f(x[dt>=dt[-1]-pd.tseries.offsets.BDay(n)]))
for h in [5,20]:
 y=[]
 for d in close.index:
  z=pd.concat([sig.loc[d],(close.shift(-h)/close-1).loc[d]],axis=1).dropna()
  if len(z)>=8 and sig.loc[d].abs().sum()>0:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): y.append(c)
 print('decay',h,f(y))
sig.index.name='date';sig.to_csv('scripts/miner_2_20310529_vix_shock_trend_signal.csv')
