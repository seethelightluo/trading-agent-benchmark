import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3000) for s in U}; D={s:d.set_index('date').sort_index() for s,d in D.items() if d is not None and len(d)>150}
c=pd.DataFrame({s:d.close for s,d in D.items()}); r=c.pct_change(); v=r.rolling(20).std()
# Contrarian trend acceleration: favor assets whose short-term risk-adjusted trend
# has lagged their medium trend, anticipating normalization. One-session lag.
sig=((c.pct_change(20)/v)-(c.pct_change(60)/v)).shift(1)
for h in [5,10,20,40,60]:
 out=[]
 for i in range(len(c)-h):
  z=pd.concat([sig.iloc[i],c.iloc[i+h]/c.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: out.append((c.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']); print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==20:
  for nm,a,b in [('2024-2026','2024-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31')]:
   x=q[(q.date>=a)&(q.date<=b)]; print('REG',nm,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(),6),round((x.ic>0).mean(),4))
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20310123_trend_acceleration_volscaled_signal.csv',index=False)
print('UNIVERSE',len(D),'DATES',len(c))
