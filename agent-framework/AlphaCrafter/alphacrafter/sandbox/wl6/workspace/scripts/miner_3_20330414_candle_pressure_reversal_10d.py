import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in A:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 D[s]=d[['open','close','high','low']]
# Candle pressure: persistent close-vs-open displacement normalized by daily range;
# reversal hypothesis uses negative rolling pressure, emphasizing exhausted directional candles.
open_=pd.concat({s:d['open'] for s,d in D.items()},axis=1); close=pd.concat({s:d['close'] for s,d in D.items()},axis=1)
high=pd.concat({s:d['high'] for s,d in D.items()},axis=1); low=pd.concat({s:d['low'] for s,d in D.items()},axis=1)
p=close.loc[:'2033-04-13']; op=open_.reindex(p.index); hi=high.reindex(p.index); lo=low.reindex(p.index)
rng=(hi-lo).replace(0,np.nan)
pressure=((p-op)/rng).rolling(10,min_periods=6).mean()
sig=-pressure
print('candle pressure reversal; instruments',len(A),'last',p.index[-1].date())
for h in [5,10,20,40]:
 f=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z)); dates.append(p.index[i])
 x=np.asarray(vals); ser=pd.Series(x,index=pd.DatetimeIndex(dates))
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.asarray(ns)/15),4),'annual',ser.groupby(ser.index.year).mean().round(5).to_dict())
