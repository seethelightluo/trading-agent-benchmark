import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,4000)
 if d is not None and len(d)>150:
  d=d.copy(); d.date=pd.to_datetime(d.date); F[s]=d.sort_values('date').drop_duplicates('date').set_index('date')
C=pd.DataFrame({s:d.close for s,d in F.items()}); R=C.pct_change(); net=C.pct_change(30); path=R.abs().rolling(30,min_periods=15).sum(); eff=(net/(path+1e-12)).clip(-1,1); vol=R.rolling(30,min_periods=15).std(); sig=(-eff/(vol*np.sqrt(30)+1e-12)).shift(1)
for h in [5,10,20,30]:
 fwd=C.shift(-h)/C-1; ics=[]; ns=[]; dates=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z)); dates.append(dt)
 a=pd.Series(ics,index=pd.to_datetime(dates)).dropna(); print('H',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),4),'hit',round((a>0).mean(),4))
 for n in [365,120]:
  q=a.tail(n); print(' recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),4),'dates',len(q))
print('assets',len(F),'coverage',round(sig.notna().sum(axis=1).div(len(F)).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_2_20330512_inverse_efficiency_signal.csv',index=False)
