import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>260:D[s]=d.set_index('date').sort_index()
c=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); m=c.pct_change(20); res=m.sub(m.mean(axis=1),axis=0)
v=pd.DataFrame({s:d.volume for s,d in D.items()}).reindex(c.index); vr=(v.rolling(20,min_periods=12).mean()/(v.rolling(60,min_periods=30).mean()+1e-12)).clip(.5,2)
sig=(-res*vr).shift(1)
for h in [5,10,20,40,60]:
 f=c.shift(-h)/c-1; q=[]
 for dt in c.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 x=pd.DataFrame(q,columns=['date','ic','n']); z=x.ic
 print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/len(D),4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 if h==20:
  for nm,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-12-31')]:
   y=x[(x.date>=a)&(x.date<=b)]; print('REG',nm,len(y),round(y.ic.mean(),6),round(y.ic.mean()/y.ic.std(ddof=1),6),round((y.ic>0).mean(),4))
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20311225_volume_confirmed_residual_reversal_signal.csv',index=False);print('UNIVERSE',len(D),'DATES',len(c),'SIGNAL_ROWS',len(out))
