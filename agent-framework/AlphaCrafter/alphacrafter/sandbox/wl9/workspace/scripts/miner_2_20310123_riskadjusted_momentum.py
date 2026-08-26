import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>250:D[s]=d.set_index('date').sort_index()
close=pd.DataFrame({s:d.close for s,d in D.items()}); r=close.pct_change()
# Medium-term trend divided by realized risk; one-day lag prevents lookahead.
sig=(r.rolling(90,min_periods=60).sum()/r.rolling(30,min_periods=20).std()).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [5,10,20,40,60]:
 a=[]
 for i in range(len(close)-h):
  z=pd.concat([sig.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append((close.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(a,columns=['date','ic','n']);print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==60:
  for nm,a1,b1 in [('2024-2026','2024-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31')]:
   x=q[(q.date>=a1)&(q.date<=b1)];print('REG',nm,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(),6))
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20310123_riskadjusted_momentum_signal.csv',index=False)
