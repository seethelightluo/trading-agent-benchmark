import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is not None and len(d)>=140: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Reversal is focused on assets with a meaningful 60d drawdown, while
# suppressing transient noise using 20d volatility normalization.
r5=P/P.shift(5)-1; hi=P.rolling(60,min_periods=40).max(); dd=P/hi-1
v20=r.rolling(20,min_periods=15).std(); raw=-r5/(v20*np.sqrt(5)+1e-12)
# drawdown gate rises smoothly from 0.5 to 2.0 as drawdown reaches -30%
gate=(0.5+(-dd).clip(0,.30)/.30*1.5).clip(.5,2.0)
sig=(raw*gate).clip(-6,6)
rows=[]
for h in [5,10,20]:
 Q=P.shift(-h)/P-1; ic=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],Q.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): ic.append(c); dates.append(dt); ns.append(len(z))
 a=np.asarray(ic); dates=pd.DatetimeIndex(dates)
 print('horizon',h,'dates',len(a),'start',dates[0].date(),'end',dates[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
 if h==10:
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20340928_drawdown_reversal_signal.csv',index=False)
  for x,y in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-09-28')]:
   z=a[(dates>=pd.Timestamp(x))&(dates<=pd.Timestamp(y))]
   if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in dates],index=dates)
  print('turnover',round(ranks.diff().abs().mean().mean(),6))
