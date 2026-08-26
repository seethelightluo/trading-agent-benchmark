import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>250: D[s]=d.set_index('date').sort_index()
close=pd.DataFrame({s:d.close for s,d in D.items()}); ret=close.pct_change()
# Recovery reversal: buy assets with deep, persistent drawdowns, but favor those beginning to recover.
peak=close.rolling(120,min_periods=80).max(); dd=close/peak-1
recovery=close/close.shift(10)-1
vol=ret.rolling(30,min_periods=20).std()
sig=(((-dd).clip(0,0.8)) * (recovery.clip(-0.2,0.2)+0.02) / vol).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [5,10,20,40,60]:
 vals=[]
 for i in range(len(close.index)-h):
  z=pd.concat([sig.iloc[i],(close.iloc[i+h]/close.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8: vals.append((close.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n'])
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==60:
  for name,a,b in [('2024-2026','2024-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31')]:
   q2=q[(q.date>=a)&(q.date<=b)]; print('REG',name,len(q2),round(q2.ic.mean(),6),round(q2.ic.mean()/q2.ic.std(),6))
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20310123_drawdown_recovery_reversal_signal.csv',index=False)
