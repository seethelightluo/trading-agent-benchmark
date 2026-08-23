import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close=pd.DataFrame({s:get_stock_daily_data(s,3000).set_index('date')['close'].astype(float) for s in U}).sort_index(); m=get_stock_daily_data('US10Y',3000).set_index('date')['close'].astype(float)
r=close.pct_change(5); mr=m.pct_change(5)
shock=(-(mr-mr.rolling(252,min_periods=60).median())).clip(lower=0).clip(upper=.05)/.05
sig=(-r).mul(shock.reindex(close.index).fillna(0),axis=0); ics=[]; ns=[]; prev=None; tos=[]
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
x=np.array([v for _,v in ics]); ds=np.array([d for d,_ in ics])
def f(a):return (float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1)),float(np.mean(a>0)),len(a))
print('dates',len(x),'avg_inst',np.mean(ns),'coverage',np.mean(ns)/15,'turnover',np.mean(tos));print('10d',f(x))
for n in [180,360]:print('recent',n,f(x[ds>=ds[-1]-pd.tseries.offsets.BDay(n)]))
for h in [5,20]:
 y=[]
 for d in close.index:
  z=pd.concat([sig.loc[d],(close.shift(-h)/close-1).loc[d]],axis=1).dropna()
  if len(z)>=8 and sig.loc[d].abs().sum()>0:y.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,f(np.array(y)))
sig.index.name='date';sig.to_csv('scripts/miner_2_20310417_us10y_downshock_reversal_signal.csv')
