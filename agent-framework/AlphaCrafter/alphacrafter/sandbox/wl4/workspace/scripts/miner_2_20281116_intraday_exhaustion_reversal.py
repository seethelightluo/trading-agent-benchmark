import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={s:get_stock_daily_data(s, days=3200) for s in U}
allrows=[]
for s,df in frames.items():
 if df is None or len(df)<80: continue
 d=df.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); c=d.close.astype(float); o=d.open.astype(float); atr=(d.high-d.low).astype(float).rolling(20).mean()
 sig=(-(c.pct_change(5)+.35*(c-o)/o)/(atr/c).replace(0,np.nan)).clip(-10,10)
 for k in [1,5,10]:
  z=pd.DataFrame({'date':d.index,'symbol':s,'sig':sig.values,'fwd':(c.shift(-k)/c-1).values}).dropna(); z['h']=k; allrows.append(z)
R=pd.concat(allrows,ignore_index=True)
print('rows',len(R),'dates',R.date.nunique(),'symbols',R.symbol.nunique())
for k in [1,5,10]:
 q=R[R.h==k]; vals=q.groupby('date').apply(lambda x:x.sig.corr(x.fwd),include_groups=False).dropna(); recent=vals.tail(250); online=vals[vals.index>=pd.Timestamp('2026-07-16')]
 print('H',k,'dates',len(vals),'avgN',q.groupby('date').size().mean(),'IC %.5f ICIR %.5f hit %.3f'%(vals.mean(),vals.mean()/vals.std(ddof=1),(vals>0).mean()),'recent %.5f %.5f'%(recent.mean(),recent.mean()/recent.std(ddof=1)),'online %.5f %.5f'%(online.mean(),online.mean()/online.std(ddof=1)))
# coverage/turnover
sigs=[]
for s,df in frames.items():
 if df is None or len(df)<80: continue
 d=df.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); c=d.close.astype(float); o=d.open.astype(float); atr=(d.high-d.low).astype(float).rolling(20).mean(); sig=(-(c.pct_change(5)+.35*(c-o)/o)/(atr/c).replace(0,np.nan)).clip(-10,10); sigs.append(sig.rename(s))
S=pd.concat(sigs,axis=1); print('coverage %.4f rank_turnover %.4f'%(S.notna().mean().mean(),S.rank(axis=1,pct=True).diff().abs().mean().mean()))
