import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(frames).sort_index().ffill(); ret=prices.pct_change()
r20=prices.pct_change(20); r60=prices.pct_change(60); vol20=ret.rolling(20).std()
med20=r20.median(axis=1); breadth=(r20>0).mean(axis=1)
gate=np.where(breadth<0.5,1.35,np.where(breadth>0.7,0.75,1.0))
sig=(r20.sub(med20,axis=0)/vol20).mul(pd.Series(gate,index=prices.index),axis=0)
sig=sig.where(r60>0,sig*0.55)
for h in [3,5,10,20]:
 fwd=prices.shift(-h)/prices-1; vals=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8: vals.append(sig.loc[dt,ok].corr(fwd.loc[dt,ok])); ns.append(ok.sum())
 a=np.asarray(vals); ic=np.nanmean(a); sd=np.nanstd(a,ddof=1)
 print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ic/sd*np.sqrt(252) if sd else np.nan:.6f} hit={(a>0).mean():.4f}')
print('rows',len(prices),'assets',len(frames),'coverage',sig.notna().sum(axis=1).mean()/len(U),'dates',len(sig))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20300207_defensive_relative_strength_signal.csv',index=False)
