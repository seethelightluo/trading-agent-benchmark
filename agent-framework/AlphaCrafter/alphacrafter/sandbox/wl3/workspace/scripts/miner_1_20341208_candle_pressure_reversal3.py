import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): D[s]=d.set_index('date')
P=pd.concat({s:x['close'] for s,x in D.items()},axis=1).sort_index()
# Reversal of signed candle pressure: persistent close-near-high buying pressure tends to mean revert.
press=pd.concat({s:((x['close']-x['open'])/(x['high']-x['low']).replace(0,np.nan)) for s,x in D.items()},axis=1).reindex(P.index)
f=(-press.rolling(3,min_periods=3).mean()).shift(1)
print('assets',len(P.columns),'dates',len(P),'range',P.index.min(),P.index.max())
for h in [1,5,10]:
 fw=P.shift(-h)/P-1; vals=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c): vals.append(c); ns.append(len(z))
   rr=f.loc[dt].rank(pct=True)
   if prev is not None: turns.append((rr-prev).abs().mean())
   prev=rr
 q=pd.Series(vals); print('horizon',h,'IC_dates',len(q),'avg_names',round(np.mean(ns),2),'coverage',round(len(q)/max(1,len(P)-h),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),4),'hit',round((q>0).mean(),4),'turnover',round(np.mean(turns),4))
 for w in [120,252]:
  x=q.tail(w); print(' recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),4),'hit',round((x>0).mean(),4))
out=f.tail(1).T.reset_index(); out.columns=['symbol','signal']; out.to_csv('scripts/miner_1_20341208_candle_pressure_reversal3_signal.csv',index=False)
