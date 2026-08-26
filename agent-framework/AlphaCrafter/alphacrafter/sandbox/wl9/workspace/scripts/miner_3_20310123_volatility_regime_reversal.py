import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>150: D[s]=d.set_index('date').sort_index()
close=pd.DataFrame({s:d.close for s,d in D.items()}); r=close.pct_change()
# Volatility-regime reversal: penalize assets whose recent realized volatility has
# expanded versus their own medium-term baseline; lagged one session.
short=r.rolling(10).std(); base=r.rolling(60).std()
sig=-(short/base.replace(0,np.nan)).shift(1)
rows=[]
for h in [5,10,20,40,60]:
 vals=[]
 for i in range(len(close.index)-h):
  z=pd.concat([sig.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append((close.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n'])
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/len(U),4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==20:
  for name,a,b in [('2024-2026','2024-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31')]:
   q2=q[(q.date>=a)&(q.date<=b)]; print('REG',name,len(q2),round(q2.ic.mean(),6),round(q2.ic.mean()/q2.ic.std(),6),round((q2.ic>0).mean(),4))
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20310123_volatility_regime_reversal_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(close))
