import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
ret=cl.pct_change(); r20=cl.pct_change(20); r60=cl.pct_change(60)
vol=ret.rolling(60).std()*np.sqrt(252)
peak=cl.rolling(252).max(); dd=(cl/peak-1).clip(-0.8,0)
# Fade recent losses, with stronger signal when loss is embedded in a deep drawdown;
# acceleration term rewards a fresh 20d loss relative to its preceding 40d path.
accel=r20-(r60-r20)/2
sig=(-accel/(vol+0.05))*(1+1.25*(-dd))
sig=sig.clip(-5,5).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for h in [10,20,40,60]:
 fwd=cl.shift(-h)/cl-1; xs=[]; ns=[]
 for dt in sig.index:
  a,b=sig.loc[dt],fwd.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: xs.append(a[ok].corr(b[ok],method='spearman')); ns.append(ok.sum())
 x=pd.Series(xs).dropna(); print('H',h,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(len(x) and x.mean(),len(x)>1 and x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)))
cov=sig.notna().sum(axis=1)/len(U); turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(); print('coverage %.6f turnover %.6f'%(cov.mean(),turn))
fwd=cl.shift(-60)/cl-1
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
 xs=[]
 for dt in sig.index[mask]:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8: xs.append(sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman'))
 x=pd.Series(xs).dropna(); print(name,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()) if len(x)>1 else 'insufficient')
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20331124_drawdown_accel_reversal_signal.csv',index=False)
