import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={};
for a in assets:
 d=get_stock_daily_data(a,days=4000)
 if d is not None and len(d): close[a]=d.set_index('date')['close']
P=pd.DataFrame(close).sort_index(); r=P.pct_change()
# candidate: smooth directional quality; medium return discounted by path noise and reversal tail
# all inputs lagged one session by constructing signal at t from data through t
sig=(P.pct_change(40)/(0.01+ r.rolling(60).std()*np.sqrt(40))) * (r.rolling(40).mean()>0).astype(float)
# replace zero in negative trend with signed smooth score, not hard filter
sig=(P.pct_change(40)/(0.01+r.rolling(60).std()*np.sqrt(40))) * (0.5+0.5*(r.rolling(40).mean()>0))
for h in [1,3,5,10,20]:
 f=sig.shift(1); y=P.pct_change(h).shift(-h); ics=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 x=np.array(ics); print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0)))
 if h==10:
  for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
   q=np.array([v for v,dt in zip(ics,f.index) if dt.year>=lo and dt.year<=hi]); print('regime',lo,hi,'n',len(q),'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)))
  for n in [63,126,252,504]:
   q=x[-n:]; print('recent',n,'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)))
 print('coverage',round(np.mean(np.array(ns)/15),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
